#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E_make_gbm_residual.py — Residual LightGBM on top of GAM, calibrated & bagged.

- Loads artifacts/make_gam.pkl
- Trains a bagged LightGBM (monotone ↓ in distance) using GAM logit as a feature
- Isotonic-calibrated via OOF
- Saves artifacts/make_bag.pkl
- Exports UI grid: ui/fg_prob_grid_distance_env_temp_wind.json
- Writes logs/make_model_report.txt

Inputs:
  artifacts/curated_fg.parquet
  artifacts/features_fg.parquet
  artifacts/make_gam.pkl

Outputs:
  artifacts/make_bag.pkl
  ui/fg_prob_grid_distance_env_temp_wind.json
  logs/make_model_report.txt
"""

from __future__ import annotations
import json, logging, pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, roc_auc_score

from A_config import Config, setup_logging, ensure_dir

DIST_MIN, DIST_MAX = 18, 68
DISTANCES = list(range(DIST_MIN, DIST_MAX + 1))
TEMP_GRID = list(range(30, 100, 5))
WIND_GRID = list(range(0, 22, 2))
ENV_OPTS  = ["indoor","outdoor"]


class GamIsoModel:
    def __init__(self, pipe, iso: IsotonicRegression, feat_cols):
        self.pipe = pipe
        self.iso = iso
        self.feat_cols = list(feat_cols)

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        # ensure all required columns exist and are numeric (no NaNs)
        for c in self.feat_cols:
            if c not in X:
                X[c] = 0
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        return X[self.feat_cols]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._X(df)
        raw = self.pipe.predict_proba(X)[:, 1]
        cal = np.clip(self.iso.predict(raw), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - cal, cal])


def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p/(1-p))

class IsoWrapper:
    def __init__(self, base_model, iso: IsotonicRegression, feat_cols: List[str]):
        self.base = base_model
        self.iso = iso
        self.feat_cols = feat_cols
    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        for c in self.feat_cols:
            if c not in Xc: Xc[c] = 0
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce").fillna(0)
        return Xc[self.feat_cols]
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = self._X(X)
        raw = self.base.predict_proba(Xc)[:, 1]
        cal = np.clip(self.iso.predict(raw), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - cal, cal])

def bag_predict_mean(bag: Dict, X: pd.DataFrame) -> np.ndarray:
    arr = [m.predict_proba(X)[:,1] for m in bag["bags"]]
    return np.mean(np.vstack(arr), axis=0)

def select_X(df: pd.DataFrame) -> pd.DataFrame:
    # base features + GAM
    need = ["distance","is_indoor","temp_F","wind_mph",
            "wind_head_mps","wind_cross_mps","air_density_ratio","altitude_m",
            "l_gam"]
    X = df.copy()
    for c in need:
        if c not in X: X[c] = 0
    X["distance"] = pd.to_numeric(X["distance"], errors="coerce")
    X = X[(X["distance"]>=DIST_MIN)&(X["distance"]<=DIST_MAX)].copy()
    X["is_indoor"] = pd.to_numeric(X["is_indoor"], errors="coerce").fillna(0).astype(int)
    for c in ["temp_F","wind_mph","wind_head_mps","wind_cross_mps","air_density_ratio","altitude_m","l_gam"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X

def fit_bag(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
            feat_cols: List[str], n_bags=24, seed=42):
    # OOF isotonic
    gkf = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups)))))
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        base = lgb.LGBMClassifier(
            objective="binary", learning_rate=0.05, n_estimators=1000,
            num_leaves=31, max_depth=6, min_data_in_leaf=60,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
            max_bin=63, n_jobs=-1, verbose=-1,
            monotone_constraints=[-1] + [0]*(len(feat_cols)-1)  # distance first, monotone ↓
        )
        base.fit(X.iloc[tr][feat_cols], y[tr])
        oof[te] = base.predict_proba(X.iloc[te][feat_cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)

    # Bagging
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    bags = []
    for i in range(n_bags):
        sel = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, np.unique(sel))
        base = lgb.LGBMClassifier(
            objective="binary", learning_rate=0.05, n_estimators=1000,
            num_leaves=31, max_depth=6, min_data_in_leaf=60,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
            max_bin=63, n_jobs=-1, verbose=-1, random_state=seed+100+i,
            monotone_constraints=[-1] + [0]*(len(feat_cols)-1)
        )
        base.fit(X.loc[mask, feat_cols], y[mask])
        bags.append(IsoWrapper(base, iso, feat_cols))
    return {"bags": bags, "feat_cols": feat_cols}

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Residual GBM on top of GAM; export UI grid.")
    ap.add_argument("--output-dir")
    ap.add_argument("--ui-dir")
    ap.add_argument("--bags", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = Config.from_args(output_dir=args.output_dir, ui_dir=args.ui_dir, log_level=args.log_level)
    setup_logging(cfg.log_level)
    log = logging.getLogger("make_gbm_resid")

    artifacts = Path(cfg.output_dir)
    ui_dir = Path(cfg.ui_dir)
    logs = ensure_dir(artifacts.parent / "logs")

    fg_feat = artifacts / "features_fg.parquet"
    fg_cur  = artifacts / "curated_fg.parquet"
    gam_fp  = artifacts / "make_gam.pkl"
    if not fg_feat.exists() or not fg_cur.exists():
        raise FileNotFoundError("Run B_ingest.py and C_features.py first.")
    if not gam_fp.exists():
        raise FileNotFoundError("Run D_make_physics_gam.py first (artifacts/make_gam.pkl).")

    Xdf = pd.read_parquet(fg_feat)
    y = pd.read_parquet(fg_cur)["fg_made"].astype(int).values
    if "game_id" in Xdf.columns:
        groups = Xdf["game_id"].astype(str).values
    else:
        groups = np.arange(len(Xdf)).astype(str)

    # Load GAM and create l_gam feature
    with gam_fp.open("rb") as f:
        gam = pickle.load(f)
        
    def _build_gam_input(df: pd.DataFrame) -> pd.DataFrame:
        g = pd.DataFrame(index=df.index)
        g["distance"] = pd.to_numeric(df.get("distance"), errors="coerce").fillna(40)
        g["is_indoor"] = pd.to_numeric(df.get("is_indoor"), errors="coerce").fillna(0).astype(int)

        # Derive temp_c if needed
        tempF = pd.to_numeric(df.get("temp_F"), errors="coerce").fillna(60.0)
        g["temp_c"] = (tempF - 60.0) / 20.0

        # Winds (fallback to 0 if not present)
        g["wind_head_mps"]  = pd.to_numeric(df.get("wind_head_mps"),  errors="coerce").fillna(0.0)
        g["wind_cross_mps"] = pd.to_numeric(df.get("wind_cross_mps"), errors="coerce").fillna(0.0)

        # Air density & altitude (fallback neutral)
        dens = pd.to_numeric(df.get("air_density_ratio"), errors="coerce").fillna(1.0)
        g["density_c"] = dens - 1.0
        g["altitude_m"] = pd.to_numeric(df.get("altitude_m"), errors="coerce").fillna(0.0)

        return g

    X_gam = _build_gam_input(Xdf)
    p_gam = gam.predict_proba(X_gam)[:, 1]
    Xdf = Xdf.copy()
    Xdf["l_gam"] = logit(p_gam)

    # Assemble features
    X = select_X(Xdf)
    y = y[X.index.values]
    groups = groups[X.index.values]

    feat_cols = ["distance","is_indoor","temp_F","wind_mph",
                 "wind_head_mps","wind_cross_mps","air_density_ratio","altitude_m",
                 "l_gam"]

    bag = fit_bag(X, y, groups, feat_cols=feat_cols, n_bags=args.bags, seed=args.seed)

    # Save final make bag
    with (artifacts / "make_bag.pkl").open("wb") as f:
        pickle.dump(bag, f)
    log.info("Saved artifacts/make_bag.pkl")

    # Report
    p = bag_predict_mean(bag, X)
    def safe_auc(yy, pp):
        try: return roc_auc_score(yy, pp)
        except Exception: return float("nan")
    with (logs / "make_model_report.txt").open("w", encoding="utf-8") as f:
        f.write("== Make Model (GAM + GBM Residual) ==\n")
        f.write(f"Rows: {len(y):,}\n")
        f.write(f"AUC:   {safe_auc(y, p):.4f}\n")
        f.write(f"Brier: {brier_score_loss(y, p):.5f}\n")
        # quick monotonic sanity
        def pred(d, is_indoor, tempF, wind):
            Xg = pd.DataFrame({
                "distance":[d], "is_indoor":[is_indoor], "temp_F":[tempF],
                "wind_mph":[wind], "wind_head_mps":[0.44704*wind if is_indoor==0 else 0.0],
                "wind_cross_mps":[0.0], "air_density_ratio":[1.0], "altitude_m":[0.0],
                "l_gam":[logit(gam.predict_proba(pd.DataFrame({
                    "distance":[d], "is_indoor":[is_indoor], "temp_F":[tempF],
                    "wind_mph":[wind], "wind_head_mps":[0.44704*wind if is_indoor==0 else 0.0],
                    "wind_cross_mps":[0.0], "air_density_ratio":[1.0], "altitude_m":[0.0]
                }))[:,1])]
            })
            return float(bag_predict_mean(bag, Xg))
        p55 = pred(55,0,60,0); p60 = pred(60,0,60,0)
        f.write(f"Neutral outdoor: p55={p55:.3f} p60={p60:.3f}  (expect ↓: {p60<=p55})\n")

    # -------- Export league grid for UI --------
    rows = []
    for env in ENV_OPTS:
        is_indoor = 1 if env == "indoor" else 0
        for temp in TEMP_GRID:
            for wind in WIND_GRID:
                # Build grid rows (headwind only for simplicity; crosswind=0)
                wind_mps = 0.44704 * wind if is_indoor==0 else 0.0
                Xg = pd.DataFrame({
                    "distance": DISTANCES,
                    "is_indoor": is_indoor,
                    "temp_F": temp,
                    "wind_mph": wind if is_indoor==0 else 0,
                    "wind_head_mps": [wind_mps]*len(DISTANCES),
                    "wind_cross_mps": [0.0]*len(DISTANCES),
                    "air_density_ratio": [1.0]*len(DISTANCES),
                    "altitude_m": [0.0]*len(DISTANCES),
                })
                # l_gam feature
                l_g = logit(gam.predict_proba(Xg)[:,1])
                Xg["l_gam"] = l_g

                # bag predictions (mean, p10, p90 across bags)
                preds = [m.predict_proba(Xg)[:,1] for m in bag["bags"]]
                P = np.vstack(preds)
                mean = P.mean(axis=0); lo = np.quantile(P, 0.10, axis=0); hi = np.quantile(P, 0.90, axis=0)
                for d, pm, plo, phi in zip(DISTANCES, mean, lo, hi):
                    rows.append({
                        "distance": int(d), "indoor_outdoor": env, "temp_F": int(temp),
                        "wind_mph": int(wind if is_indoor==0 else 0),
                        "prob_mean": round(float(pm), 6),
                        "prob_p10":  round(float(plo), 6),
                        "prob_p90":  round(float(phi), 6),
                    })
    payload = {
        "meta": {
            "note": "GAM baseline + monotone LightGBM residual; isotonic-calibrated; wind is headwind only outdoors.",
            "distances": [DIST_MIN, DIST_MAX],
            "temps": TEMP_GRID, "winds": WIND_GRID, "env": ENV_OPTS,
            "n_bags": int(args.bags)
        },
        "grid": rows
    }
    out_json = ui_dir / "fg_prob_grid_distance_env_temp_wind.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    log.info("Wrote %s", out_json)
    log.info("Done.")

if __name__ == "__main__":
    main()
