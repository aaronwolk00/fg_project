#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_attempt_model.py — Attempt propensity model (4th down, FG vs not).

Inputs (from 01_ingest / 02_features in cfg.output_dir):
  - curated_attempt.parquet
  - features_attempt.parquet

Outputs:
  - artifacts/attempt_bag.pkl
  - ui/fg_attempt_grid_distance_env_context.json
  - logs/attempt_model_report.txt
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, roc_auc_score

from A_config import Config, setup_logging, ensure_dir

# ------------------------- Grid & bins -------------------------
DIST_MIN, DIST_MAX = 18, 68
DISTANCES = list(range(DIST_MIN, DIST_MAX + 1))

SCORE_BINS = ["trail", "close", "lead"]
TIME_BINS  = ["early", "mid", "late"]
YTG_BINS   = ["short", "med", "long"]
REP_SCORE  = {"trail": -7, "close": 0, "lead": +7}
REP_TIME   = {"early": 1200, "mid": 600, "late": 120}
REP_YTG    = {"short": 2, "med": 5, "long": 9}

# ------------------------- Helpers -------------------------

def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

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

def fit_bag(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
            feat_cols: List[str], n_bags: int = 14, seed: int = 42):
    rng = np.random.RandomState(seed)
    bags = []

    # Out-of-fold isotonic
    gkf = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups)))))
    oof_pred = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        base = lgb.LGBMClassifier(
            objective="binary", learning_rate=0.05, n_estimators=800,
            num_leaves=31, max_depth=6, min_data_in_leaf=60,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
            verbose=-1, max_bin=63, n_jobs=-1
        )
        base.fit(X.iloc[tr][feat_cols], y[tr])
        oof_pred[te] = base.predict_proba(X.iloc[te][feat_cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_pred, y)

    # Bagging on random group subsets
    uniq = np.unique(groups)
    for i in range(n_bags):
        sel = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, np.unique(sel))
        base = lgb.LGBMClassifier(
            objective="binary", learning_rate=0.05, n_estimators=800,
            num_leaves=31, max_depth=6, min_data_in_leaf=60,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
            verbose=-1, max_bin=63, n_jobs=-1, random_state=seed + 100 + i
        )
        base.fit(X.loc[mask, feat_cols], y[mask])
        bags.append(IsoWrapper(base, iso, feat_cols))
    return {"bags": bags, "feat_cols": feat_cols}

def bag_mean_prob(bag: Dict, X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(X)[:, 1] for m in bag["bags"]]
    return np.mean(np.vstack(preds), axis=0)

# ------------------------- Main -------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train attempt model and export league grid.")
    ap.add_argument("--output-dir")
    ap.add_argument("--ui-dir")
    ap.add_argument("--bags", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = Config.from_args(output_dir=args.output_dir, ui_dir=args.ui_dir, log_level=args.log_level)
    setup_logging(cfg.log_level)
    log = logging.getLogger("06_attempt_model")

    artifacts = Path(cfg.output_dir)
    ui_dir = Path(cfg.ui_dir)
    logs_dir = ensure_dir(artifacts.parent / "logs")

    att_cur = artifacts / "curated_attempt.parquet"
    att_feat = artifacts / "features_attempt.parquet"
    for fp in [att_cur, att_feat]:
        if not fp.exists():
            raise FileNotFoundError(f"Missing {fp}. Run 01_ingest.py and 02_features.py first.")

    att = pd.read_parquet(att_feat)
    y = pd.read_parquet(att_cur)["attempt_fg"].astype(int).values

    # Features (robust to missing)
    feat_cols = ["distance","is_indoor","score_diff","half_sec","ydstogo",
                 "wind_head_mps","wind_cross_mps","air_density_ratio","altitude_m"]
    for c in feat_cols:
        if c not in att.columns: att[c] = 0
    X = att.copy()
    X["score_diff"] = X["score_diff"].clip(-21, 21)
    X["half_sec"] = pd.to_numeric(X["half_sec"], errors="coerce").fillna(900)
    X["ydstogo"] = pd.to_numeric(X["ydstogo"], errors="coerce").fillna(0)
    X["is_indoor"] = pd.to_numeric(X["is_indoor"], errors="coerce").fillna(0).astype(int)
    for c in ["wind_head_mps","wind_cross_mps","air_density_ratio","altitude_m","distance"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    groups = pd.read_parquet(att_cur)["game_id"].astype(str).values

    bag = fit_bag(X, y, groups, feat_cols=feat_cols, n_bags=args.bags, seed=args.seed)

    # Save model
    with (artifacts / "attempt_bag.pkl").open("wb") as f:
        pickle.dump(bag, f)

    # Report
    p = bag_mean_prob(bag, X)
    with (logs_dir / "attempt_model_report.txt").open("w", encoding="utf-8") as f:
        def safe_auc(y_true, p):
            try: return roc_auc_score(y_true, p)
            except Exception: return float("nan")
        f.write("== Attempt Model Report ==\n")
        f.write(f"Rows: {len(y):,}\n")
        f.write(f"AUC:   {safe_auc(y, p):.4f}\n")
        f.write(f"Brier: {brier_score_loss(y, p):.5f}\n")

    # Export league grid
    rows = []
    for env in ["indoor","outdoor"]:
        is_indoor = 1 if env == "indoor" else 0
        for score in ["trail","close","lead"]:
            for t in ["early","mid","late"]:
                for ytg in ["short","med","long"]:
                    Xg = pd.DataFrame({
                        "distance": DISTANCES,
                        "is_indoor": is_indoor,
                        "score_diff": REP_SCORE[score],
                        "half_sec": REP_TIME[t],
                        "ydstogo": REP_YTG[ytg],
                        "wind_head_mps": 0.0,
                        "wind_cross_mps": 0.0,
                        "air_density_ratio": 1.0,
                        "altitude_m": 0.0,
                    })
                    probs = bag_mean_prob(bag, Xg)
                    for d, pr in zip(DISTANCES, probs):
                        rows.append({
                            "distance": int(d),
                            "indoor_outdoor": env,
                            "score_bin": score, "time_bin": t, "ytg_bin": ytg,
                            "prob_attempt_mean": round(float(pr), 6)
                        })
    payload = {"meta": {
        "note": "Attempt propensity; bagged LGBM + isotonic. Wind set to 0; density=1; altitude=0.",
        "distances": [DIST_MIN, DIST_MAX],
        "env": ["indoor","outdoor"],
        "score_bins": SCORE_BINS, "time_bins": TIME_BINS, "ytg_bins": YTG_BINS,
        "rep_values": {"score": REP_SCORE, "time_half_sec": REP_TIME, "ydstogo": REP_YTG}
    }, "grid": rows}
    out_json = ui_dir / "fg_attempt_grid_distance_env_context.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    log.info("Wrote %s", out_json)

    log.info("Done.")

if __name__ == "__main__":
    main()
