#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FG Pipeline — Professional Upgrade (SECTION 2/2)

This module consumes the curated artifacts from SECTION 1 and delivers
commercial‑grade modeling and exports for the Pro Coach UI:

Outputs → <OUTPUT_DIR>
  - fg_prob_grid_distance_env_temp_wind.json    # league make prob grid (env × temp × wind × distance)
  - kicker_deltas_logit_banded.json             # kicker distance‑band deltas (recency‑weighted, Bayesian)
  - fg_attempt_grid_distance_env_context.json   # league attempt propensity grid (context × distance)
  - attempt_sensitivity.json                    # coupling: Attempt ~ A + C*logit(P_base) + B*Δlogit(make)
  - training_report.txt                         # metrics, diagnostics, feature importances

Highlights
- **Make model**: Bagged LightGBM with **monotone constraint** in distance and
  out‑of‑fold **Isotonic** calibration. Bagging is **group‑bootstrap by game** to
  suppress leak and stabilize calibration across seasons.
- **Attempt model**: Bagged LightGBM + Isotonic on 4th‑down rows with compact
  context features (distance, indoor, score, time-in-half, ydstogo).
- **Uncertainty**: p10/p90 from the empirical distribution across bags at each
  gridpoint (well‑calibrated when bags are diverse). Diagnostics report includes
  reliability summaries.
- **Kicker deltas**: Recency‑weighted empirical Bayes (Beta‑Binomial) by band
  with weaker prior in 60+ yard bucket. Smooth band blending at boundaries.
- **Coupling**: Learn A,B,C so Attempt% rises with a kicker’s **logit advantage**
  vs league make at that distance & weather.

Run example (matching Section 1 defaults):
  python fg_pipeline_section2.py \
      --pbp-root "C:/Users/awolk/Documents/NFELO/Other Data/nflverse/pbp" \
      --output-dir "C:/Users/awolk/Documents/NFELO/Other Data/nflverse/pbp/model_outputs"

SECTION 1 must be run first so the curated parquet files exist in OUTPUT_DIR.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

# ----------------------------- Config ------------------------------------

@dataclass
class Config:
    pbp_root: Path
    output_dir: Path
    # model + export settings
    n_bags_make: int = 24
    n_bags_attempt: int = 14
    early_stop: int = 60
    max_folds: int = 5
    random_seed: int = 42
    # export grids
    dist_min: int = 18
    dist_max: int = 68
    temp_grid: List[int] = tuple(range(30, 101, 5))  # °F
    wind_grid: List[int] = tuple(range(0,  23, 2))   # mph
    env_opts: List[str] = ("indoor", "outdoor")

    # kicker priors by distance band
    recency_half_life_days: float = 360.0
    prior_strength_by_band: Dict[str, float] = None

    def __post_init__(self):
        if self.prior_strength_by_band is None:
            self.prior_strength_by_band = {
                "short": 150.0,  # <=39
                "mid":   120.0,  # 40–49
                "long":   80.0,  # 50–59
                "xlong":  20.0,  # 60+
            }


CUR_FG = "curated_fg.parquet"
CUR_ATT = "curated_attempt.parquet"
ROSTER_JSON = "nfl_kicker_roster.json"

GRID_JSON = "fg_prob_grid_distance_env_temp_wind.json"
KICKERS_JSON_BANDED = "kicker_deltas_logit_banded.json"
ATTEMPT_JSON = "fg_attempt_grid_distance_env_context.json"
ATTEMPT_SENS_JSON = "attempt_sensitivity.json"
REPORT_TXT = "training_report.txt"


# ----------------------------- Logging -----------------------------------

def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "section2.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Logging to %s", log_path)


# ----------------------------- Utils -------------------------------------

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1.0 - p))


def dist_band(d: int) -> str:
    if d <= 39:
        return "short"
    if d <= 49:
        return "mid"
    if d <= 59:
        return "long"
    return "xlong"


# ----------------------- Modeling Primitives ------------------------------

def lgbm_base(mono: Optional[List[int]] = None, seed: int = 123, n_jobs: Optional[int] = None) -> lgb.LGBMClassifier:
    params = dict(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=6,
        min_data_in_leaf=80,
        reg_alpha=0.2,
        reg_lambda=0.4,
        subsample=0.8,
        colsample_bytree=0.8,
        max_bin=127,
        n_jobs=n_jobs or max(1, (os.cpu_count() or 4) - 1),
        verbose=-1,
    )
    if mono is not None:
        params["monotone_constraints"] = mono
        params["monotone_constraints_method"] = "basic"
    return lgb.LGBMClassifier(**params)


class IsoWrapper:
    """Wrapper with OOF isotonic for calibrated probabilities."""
    def __init__(self, base_model: lgb.LGBMClassifier, iso: IsotonicRegression, feat_cols: List[str]):
        self.base = base_model
        self.iso = iso
        self.feat_cols = feat_cols

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        for c in self.feat_cols:
            if c not in Xc.columns:
                Xc[c] = 0
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce").fillna(0)
        return Xc[self.feat_cols]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = self._align(X)
        p = self.base.predict_proba(Xc)[:, 1]
        pc = np.clip(self.iso.predict(p), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - pc, pc])


def fit_bag(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    feat_cols: List[str],
    n_bags: int,
    base_seed: int,
    early_stopping_rounds: int,
    max_folds: int,
    monotone_on: Optional[List[int]] = None,
) -> Dict:
    rng = np.random.RandomState(base_seed)
    bags: List[IsoWrapper] = []

    mono = None
    if monotone_on is not None:
        mono = [0] * len(feat_cols)
        for idx in monotone_on:
            if 0 <= idx < len(feat_cols):
                mono[idx] = -1  # distance monotone decreasing

    unique_groups = np.unique(groups_all)

    def fit_with_oof_isotonic(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int) -> Optional[IsoWrapper]:
        uniq = np.unique(groups)
        if len(uniq) < 2:
            return None
        n_splits = min(max_folds, max(2, len(uniq)))
        gkf = GroupKFold(n_splits=n_splits)
        oof_pred = np.zeros(len(X), dtype=float)
        for tr, te in gkf.split(X, y, groups):
            if y[tr].min() == y[tr].max():
                oof_pred[te] = y[tr].mean()
                continue
            m = lgbm_base(mono, seed + 17)
            m.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[te], y[te])],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            oof_pred[te] = m.predict_proba(X.iloc[te])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_pred, y)

        # Holdout by groups for early stopping on the final fit
        uniq_groups = np.unique(groups)
        take = max(1, len(uniq_groups) // 5)
        val_groups = set(uniq_groups[-take:])
        val_mask = np.array([g in val_groups for g in groups])
        tr_mask = ~val_mask

        m_full = lgbm_base(mono, seed + 33)
        if y[tr_mask].min() == y[tr_mask].max():
            m_full.fit(X, y)
        else:
            m_full.fit(
                X.iloc[tr_mask], y[tr_mask],
                eval_set=[(X.iloc[val_mask], y[val_mask])],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
        return IsoWrapper(m_full, iso, feat_cols=feat_cols)

    attempts = 0
    while len(bags) < n_bags and attempts < n_bags * 4:
        attempts += 1
        seed = int(base_seed + 1000 + attempts)
        sel_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sel_groups = np.unique(sel_groups)
        mask = np.isin(groups_all, sel_groups)
        model_obj = fit_with_oof_isotonic(X_all.loc[mask], y_all[mask], groups_all[mask], seed)
        if model_obj is None:
            continue
        bags.append(model_obj)
    return {"bags": bags, "feat_cols": feat_cols}


def bag_predict_proba(bag: Dict, X: pd.DataFrame) -> np.ndarray:
    preds = []
    for m in bag["bags"]:
        preds.append(m.predict_proba(X)[:, 1])
    return np.vstack(preds)


# --------------------- Kicker Deltas (recency + Bayes) --------------------

def exp_weight(age_days: float, half_life_days: float) -> float:
    return 0.5 ** (max(age_days, 0.0) / max(half_life_days, 1.0))


def beta_posterior(alpha0: float, beta0: float, s_eff: float, n_eff: float) -> Tuple[float, float]:
    alpha = alpha0 + s_eff
    beta = beta0 + max(n_eff - s_eff, 0.0)
    p = alpha / (alpha + beta)
    var_p = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    eps = 1e-9
    var_logit = var_p / (max(p, eps) ** 2 * max(1 - p, eps) ** 2)
    se_logit = float(np.sqrt(max(var_logit, 1e-12)))
    return float(p), se_logit


def compute_kicker_deltas(df_fg: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    assert {"distance", "fg_made", "game_date", "kicker_id", "kicker_name", "season_year"}.issubset(df_fg.columns)
    df = df_fg.copy()
    df["band"] = df["distance"].astype(int).apply(dist_band)

    today = pd.to_datetime(str(int(df["season_year"].max())) + "-12-31")
    if df["game_date"].notna().any():
        today = pd.to_datetime(df["game_date"].max())
    age_days = (today - df["game_date"]).dt.days.astype(float)
    df["w"] = age_days.apply(lambda d: exp_weight(d, cfg.recency_half_life_days))

    league = (
        df.groupby("band").apply(lambda g: pd.Series({"n_eff": g["w"].sum(), "s_eff": (g["w"] * g["fg_made"]).sum()})).reset_index()
    )
    league["p_league"] = league["s_eff"] / league["n_eff"].replace(0, np.nan)
    league["p_league"] = league["p_league"].fillna(0.5)

    prior = {}
    for _, r in league.iterrows():
        band = r["band"]
        pL = float(r["p_league"])
        strength = cfg.prior_strength_by_band.get(band, 100.0)
        prior[band] = {"p_league": pL, "alpha0": strength * pL, "beta0": strength * (1 - pL)}

    agg = (
        df.groupby(["kicker_id", "kicker_name", "band"]).apply(
            lambda g: pd.Series({"n_eff": g["w"].sum(), "s_eff": (g["w"] * g["fg_made"]).sum()})
        ).reset_index()
    )

    rows = []
    def _logit(p: float) -> float:
        p = max(min(p, 1 - 1e-9), 1e-9)
        return float(math.log(p / (1 - p)))

    for _, row in agg.iterrows():
        kid = str(row["kicker_id"]) ; kname = row["kicker_name"] ; band = row["band"]
        n_eff = float(row["n_eff"]) ; s_eff = float(row["s_eff"]) ; info = prior.get(band)
        p_post, se_logit = beta_posterior(info["alpha0"], info["beta0"], s_eff, n_eff)
        delta = _logit(p_post) - _logit(info["p_league"])
        rows.append({"kicker_id": kid, "kicker_name": kname, "band": band, "n_eff": n_eff,
                     "delta_logit": float(delta), "se": float(se_logit)})

    out = pd.DataFrame(rows)
    packed = []
    for kid, grp in out.groupby("kicker_id"):
        kname = grp["kicker_name"].iloc[0]
        eff = float(grp["n_eff"].sum())
        bands = {r["band"]: {"delta_logit": float(r["delta_logit"]), "se": float(r["se"]), "n_eff": float(r["n_eff"]) } for _, r in grp.iterrows()}
        packed.append({"kicker_id": kid, "kicker_name": kname, "attempts_eff": eff, "bands": bands})
    return pd.DataFrame(packed)


def blend_kicker_shift(bands: Dict[str, Dict[str, float]], d: int) -> Tuple[float, float]:
    """Smooth blend of band deltas at boundaries (±2 yards)."""
    BLEND_W = 2
    bounds = {"short": 39, "mid": 49, "long": 59}
    def weights(d: int) -> Dict[str, float]:
        if d <= bounds["short"] - BLEND_W:
            return {"short": 1.0}
        if d < bounds["short"] + BLEND_W:
            t = (d - (bounds["short"] - BLEND_W)) / (2 * BLEND_W)
            return {"short": 1 - t, "mid": t}
        if d <= bounds["mid"] - BLEND_W:
            return {"mid": 1.0}
        if d < bounds["mid"] + BLEND_W:
            t = (d - (bounds["mid"] - BLEND_W)) / (2 * BLEND_W)
            return {"mid": 1 - t, "long": t}
        if d <= bounds["long"] - BLEND_W:
            return {"long": 1.0}
        if d < bounds["long"] + BLEND_W:
            t = (d - (bounds["long"] - BLEND_W)) / (2 * BLEND_W)
            return {"long": 1 - t, "xlong": t}
        return {"xlong": 1.0}
    w = weights(d)
    delta, var_sum = 0.0, 0.0
    for b, ww in w.items():
        if b in bands:
            delta += ww * float(bands[b].get("delta_logit", 0.0))
            se = float(bands[b].get("se", 0.0))
            var_sum += (ww * ww) * (se * se)
    return float(delta), float(math.sqrt(var_sum))


# ----------------------------- Exports -----------------------------------

def export_make_grid(bag: Dict, cfg: Config, out_path: Path) -> None:
    rows: List[Dict] = []
    distances = list(range(cfg.dist_min, cfg.dist_max + 1))
    for env in cfg.env_opts:
        is_indoor = 1 if env == "indoor" else 0
        for temp in cfg.temp_grid:
            for wind in cfg.wind_grid:
                w = 0 if is_indoor == 1 else wind
                X = pd.DataFrame({
                    "distance": distances,
                    "is_indoor": [is_indoor] * len(distances),
                    "temp_F": [temp] * len(distances),
                    "wind_mph": [w] * len(distances),
                    "season_year": [2024] * len(distances),
                })
                P = bag_predict_proba(bag, X)  # shape: (n_bags, n_points)
                mean = P.mean(axis=0)
                lo = np.quantile(P, 0.10, axis=0)
                hi = np.quantile(P, 0.90, axis=0)
                for i, d in enumerate(distances):
                    rows.append({
                        "distance": d,
                        "indoor_outdoor": env,
                        "temp_F": int(temp),
                        "wind_mph": int(w),
                        "prob_mean": round(float(mean[i]), 6),
                        "prob_p10": round(float(lo[i]), 6),
                        "prob_p90": round(float(hi[i]), 6),
                    })
    payload = {
        "meta": {
            "note": "Bagged LGBM (monotone distance) + OOF-Isotonic; p10/p90 across bag",
            "distances": [cfg.dist_min, cfg.dist_max],
            "temps": list(cfg.temp_grid),
            "winds": list(cfg.wind_grid),
            "env": list(cfg.env_opts),
            "n_bags": len(bag["bags"]),
        },
        "grid": rows,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    logging.info("[OK] Make grid → %s", out_path)


def export_attempt_grid(bag_att: Dict, cfg: Config, out_path: Path) -> None:
    rows: List[Dict] = []
    distances = list(range(cfg.dist_min, cfg.dist_max + 1))

    SCORE_BINS = ["trail", "close", "lead"]
    TIME_BINS = ["early", "mid", "late"]
    YTG_BINS = ["short", "med", "long"]
    REP_SCORE = {"trail": -7, "close": 0, "lead": 7}
    REP_TIME = {"early": 1200, "mid": 600, "late": 120}
    REP_YTG = {"short": 2, "med": 5, "long": 9}

    for env in cfg.env_opts:
        is_indoor = 1 if env == "indoor" else 0
        for score_bin in SCORE_BINS:
            for time_bin in TIME_BINS:
                for ybin in YTG_BINS:
                    score = REP_SCORE[score_bin]
                    halfs = REP_TIME[time_bin]
                    ytg = REP_YTG[ybin]
                    X = pd.DataFrame({
                        "distance": distances,
                        "is_indoor": [is_indoor] * len(distances),
                        "season_year": [2024] * len(distances),
                        "score_diff": [score] * len(distances),
                        "half_sec": [halfs] * len(distances),
                        "ydstogo": [ytg] * len(distances),
                    })
                    P = bag_predict_proba(bag_att, X)
                    mean = P.mean(axis=0)
                    lo = np.quantile(P, 0.10, axis=0)
                    hi = np.quantile(P, 0.90, axis=0)
                    for i, d in enumerate(distances):
                        rows.append({
                            "distance": d,
                            "indoor_outdoor": env,
                            "score_bin": score_bin,
                            "time_bin": time_bin,
                            "ytg_bin": ybin,
                            "prob_attempt_mean": round(float(mean[i]), 6),
                            "prob_attempt_p10": round(float(lo[i]), 6),
                            "prob_attempt_p90": round(float(hi[i]), 6),
                        })
    payload = {
        "meta": {
            "note": "4th-down attempt propensity; bagged LGBM + OOF-Isotonic; bins: score/time/ytg",
            "distances": [cfg.dist_min, cfg.dist_max],
            "env": list(cfg.env_opts),
            "score_bins": ["trail", "close", "lead"],
            "time_bins": ["early", "mid", "late"],
            "ytg_bins": ["short", "med", "long"],
            "rep_values": {
                "score": {"trail": -7, "close": 0, "lead": 7},
                "time_half_sec": {"early": 1200, "mid": 600, "late": 120},
                "ydstogo": {"short": 2, "med": 5, "long": 9},
            },
            "n_bags": len(bag_att["bags"]),
        },
        "grid": rows,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    logging.info("[OK] Attempt grid → %s", out_path)


# --------------------------- Diagnostics ---------------------------------

def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def reliability_bins(y: np.ndarray, p: np.ndarray, k: int = 10) -> List[Tuple[float, float, int]]:
    q = np.quantile(p, np.linspace(0, 1, k + 1))
    q[0], q[-1] = 0.0, 1.0
    idx = np.digitize(p, q[1:-1], right=False)
    out = []
    for b in range(k):
        m = idx == b
        if m.sum() == 0:
            out.append((float("nan"), float("nan"), 0))
        else:
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


# ------------------------------ Main -------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="FG Pipeline — Section 2/2 (modeling + exports)")
    parser.add_argument("--pbp-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bags-make", type=int, default=24)
    parser.add_argument("--bags-attempt", type=int, default=14)
    args = parser.parse_args(argv)

    pbp_default = Path(r"C:/Users/awolk/Documents/NFELO/Other Data/nflverse/pbp")
    out_default = pbp_default / "model_outputs"
    cfg = Config(
        pbp_root=Path(args.pbp_root or pbp_default),
        output_dir=Path(args.output_dir or out_default),
        n_bags_make=int(args.bags_make),
        n_bags_attempt=int(args.bags_attempt),
        random_seed=int(args.seed),
    )

    setup_logging(cfg.output_dir)
    seed_everything(cfg.random_seed)

    # 1) Load curated datasets from Section 1
    fg_path = cfg.output_dir / CUR_FG
    att_path = cfg.output_dir / CUR_ATT
    roster_path = cfg.output_dir / ROSTER_JSON

    if not fg_path.exists() or not att_path.exists():
        raise FileNotFoundError("Curated files not found. Run Section 1 first.")

    df_fg = pd.read_parquet(fg_path)
    df_att = pd.read_parquet(att_path)
    roster = {}
    if roster_path.exists():
        roster = json.loads(roster_path.read_text(encoding="utf-8"))

    logging.info("Loaded curated_fg=%d, curated_attempt=%d", len(df_fg), len(df_att))

    # 2) MAKE MODEL (league)
    feat_make = ["distance", "is_indoor", "temp_F", "wind_mph", "season_year"]
    Xm = df_fg[feat_make].copy()
    ym = df_fg["fg_made"].astype(int).values
    gm = df_fg["game_id"].astype(str).values

    bag_make = fit_bag(
        Xm, ym, gm, feat_cols=feat_make,
        n_bags=cfg.n_bags_make,
        base_seed=cfg.random_seed,
        early_stopping_rounds=cfg.early_stop,
        max_folds=cfg.max_folds,
        monotone_on=[0],  # distance monotone decreasing
    )

    # quick eval (OOF not kept; do bag mean on full set)
    pm = bag_predict_proba(bag_make, Xm).mean(axis=0)

    # 3) KICKER DELTAS (recency + Bayes)
    kicker_bands_df = compute_kicker_deltas(df_fg, cfg)

    # Helper map
    KMAP = {row["kicker_id"]: row["bands"] for _, row in kicker_bands_df.iterrows()}

    def kicker_shift_for_row(kid: Optional[str], d: int) -> float:
        if not isinstance(kid, str) or kid not in KMAP:
            return 0.0
        delta, _se = blend_kicker_shift(KMAP[kid], d)
        return float(delta)

    # 4) ATTEMPT MODEL (league context)
    feat_att = ["distance", "is_indoor", "season_year", "score_diff", "half_sec", "ydstogo"]
    Xa = df_att[feat_att].copy()
    ya = df_att["attempt_fg"].astype(int).values
    ga = df_att["game_id"].astype(str).values

    bag_att = fit_bag(
        Xa, ya, ga, feat_cols=feat_att,
        n_bags=cfg.n_bags_attempt,
        base_seed=cfg.random_seed + 777,
        early_stopping_rounds=cfg.early_stop,
        max_folds=cfg.max_folds,
        monotone_on=None,
    )

    pa = bag_predict_proba(bag_att, Xa).mean(axis=0)

    # 5) COUPLING: learn A,B,C for Attempt ~ A + C*logit(P_base) + B*Δlogit(make)
    # compute league make prob row-wise at ATT rows (env/weather-aware)
    Xm_row = pd.DataFrame({
        "distance": df_att["distance"].values,
        "is_indoor": df_att["is_indoor"].values,
        "temp_F": df_att.get("temp_F", pd.Series([65] * len(df_att))).values,
        "wind_mph": np.where(df_att["is_indoor"].values == 1, 0, df_att.get("wind_mph", pd.Series([5] * len(df_att))).values),
        "season_year": df_att.get("season_year", pd.Series([2024] * len(df_att))).values,
    })
    Pm_league = bag_predict_proba(bag_make, Xm_row).mean(axis=0)
    lPm_league = logit(Pm_league)

    kid_series = df_att.get("kicker_id_game", pd.Series([None] * len(df_att)))
    deltas = np.array([kicker_shift_for_row(k, d) for k, d in zip(kid_series, df_att["distance"])])
    Pm_kicker = sigmoid(lPm_league + deltas)
    lPm_kicker = logit(Pm_kicker)
    delta_logit_make = lPm_kicker - lPm_league

    Pa_base = bag_predict_proba(bag_att, Xa).mean(axis=0)
    lPa_base = logit(Pa_base)

    X_lr = np.column_stack([lPa_base, delta_logit_make])
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e6)
    lr.fit(X_lr, ya)
    A = float(lr.intercept_[0])
    C = float(lr.coef_[0, 0])
    B = float(lr.coef_[0, 1])

    # 6) Exports
    export_make_grid(bag_make, cfg, cfg.output_dir / GRID_JSON)
    kicker_bands_df.to_json(cfg.output_dir / KICKERS_JSON_BANDED, orient="records")
    logging.info("[OK] Kicker banded JSON → %s", cfg.output_dir / KICKERS_JSON_BANDED)

    export_attempt_grid(bag_att, cfg, cfg.output_dir / ATTEMPT_JSON)

    sens = {
        "note": "logit(Attempt) = A + C*logit(P_attempt_base) + B*(logit(P_make_kicker)-logit(P_make_league))",
        "A": A,
        "B": B,
        "C": C,
    }
    with (cfg.output_dir / ATTEMPT_SENS_JSON).open("w", encoding="utf-8") as f:
        json.dump(sens, f)
    logging.info("[OK] Attempt sensitivity → %s  (A=%.3f, B=%.3f, C=%.3f)", cfg.output_dir / ATTEMPT_SENS_JSON, A, B, C)

    # 7) Reporting
    report_path = cfg.output_dir / REPORT_TXT
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n== Section 2 Report ==\n")
        # Make model
        f.write("\n[Make model]\n")
        f.write(f"AUC:   {roc_auc_score(ym, pm):.4f}\n")
        f.write(f"Brier: {brier_score_loss(ym, pm):.5f}\n")
        rel = reliability_bins(ym, pm, k=10)
        f.write("Reliability (mean_p, event_rate, n):\n")
        for mp, er, n in rel:
            f.write(f"  {mp:.3f}\t{er:.3f}\t(n={n})\n")
        # Attempt model
        f.write("\n[Attempt model]\n")
        f.write(f"AUC:   {safe_auc(ya, pa):.4f}\n")
        f.write(f"Brier: {brier_score_loss(ya, pa):.5f}\n")
        rel_a = reliability_bins(ya, pa, k=10)
        f.write("Reliability (mean_p, event_rate, n):\n")
        for mp, er, n in rel_a:
            f.write(f"  {mp:.3f}\t{er:.3f}\t(n={n})\n")
        # Coupling
        f.write("\n[Coupling]\n")
        f.write(f"A: {A:.3f}, B: {B:.3f}, C: {C:.3f}\n")

    logging.info("[DONE] Exports + report written to %s", cfg.output_dir)


if __name__ == "__main__":
    main()
