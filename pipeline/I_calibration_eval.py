#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_calibration_eval.py — Reliability & diagnostics for make and attempt models.

Inputs:
  - artifacts/features_fg.parquet
  - artifacts/curated_fg.parquet
  - artifacts/features_attempt.parquet
  - artifacts/curated_attempt.parquet
  - artifacts/attempt_bag.pkl
  - artifacts/make_bag.pkl (optional; else fallback trained here like 07)

Outputs (to logs/):
  - make_calibration.json
  - attempt_calibration.json
  - calibration_report.txt
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from A_config import Config, setup_logging, ensure_dir

# --------------------- Helpers ---------------------

def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

class IsoWrapper:
    def __init__(self, base_model, iso: IsotonicRegression, feat_cols):
        self.base = base_model; self.iso = iso; self.feat_cols = feat_cols
    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        for c in self.feat_cols:
            if c not in Xc: Xc[c] = 0
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce").fillna(0)
        return Xc[self.feat_cols]
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base.predict_proba(self._X(X))[:, 1]
        cal = np.clip(self.iso.predict(raw), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - cal, cal])

def fit_make_fallback(features_fg_fp: Path):
    df = pd.read_parquet(features_fg_fp)
    y = pd.read_parquet(features_fg_fp.parent / "curated_fg.parquet")["fg_made"].astype(int).values
    feat_cols = ["distance","is_indoor","temp_F","wind_mph","air_density_ratio","wind_head_mps","wind_cross_mps"]
    for c in feat_cols:
        if c not in df.columns: df[c] = 0
        df["is_indoor"] = pd.to_numeric(df["is_indoor"], errors="coerce").fillna(0).astype(int)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    groups = df.get("game_id", pd.Series(np.arange(len(df)))).astype(str).values
    gkf = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups)))))
    oof = np.zeros(len(y))
    for tr, te in gkf.split(df, y, groups):
        base = lgb.LGBMClassifier(
            objective="binary", learning_rate=0.05, n_estimators=900, num_leaves=31, max_depth=6,
            min_data_in_leaf=60, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
            verbose=-1, max_bin=63, n_jobs=-1,
            monotone_constraints=[-1, 0, 0, 0, 0, 0, 0]
        )
        base.fit(df.iloc[tr][feat_cols], y[tr])
        oof[te] = base.predict_proba(df.iloc[te][feat_cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    base_full = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, n_estimators=900, num_leaves=31, max_depth=6,
        min_data_in_leaf=60, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
        verbose=-1, max_bin=63, n_jobs=-1,
        monotone_constraints=[-1, 0, 0, 0, 0, 0, 0]
    )
    base_full.fit(df[feat_cols], y)
    return {"bags": [IsoWrapper(base_full, iso, feat_cols)], "feat_cols": feat_cols}

def bag_mean_prob(bag: Dict, X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(X)[:, 1] for m in bag["bags"]]
    return np.mean(np.vstack(preds), axis=0)

def reliability(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict:
    df = pd.DataFrame({"y": y_true, "p": p})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    out = []
    for cat, g in df.groupby("bin"):
        if len(g) < 10: continue
        out.append({
            "bin": str(cat),
            "n": int(len(g)),
            "p_mean": float(g["p"].mean()),
            "y_rate": float(g["y"].mean())
        })
    return {"bins": out}

# --------------------- Main ---------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Calibration / reliability diagnostics for make & attempt.")
    ap.add_argument("--output-dir")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = Config.from_args(output_dir=args.output_dir, log_level=args.log_level)
    setup_logging(cfg.log_level)
    log = logging.getLogger("08_calibration_eval")

    artifacts = Path(cfg.output_dir)
    logs_dir = ensure_dir(artifacts.parent / "logs")

    # --- Make ---
    fg_feat = artifacts / "features_fg.parquet"
    fg_cur  = artifacts / "curated_fg.parquet"
    if not fg_feat.exists() or not fg_cur.exists():
        raise FileNotFoundError("Need features_fg.parquet and curated_fg.parquet. Run 01 & 02.")

    try:
        with (artifacts / "make_bag.pkl").open("rb") as f:
            make_bag = pickle.load(f)
    except FileNotFoundError:
        make_bag = fit_make_fallback(fg_feat)

    df_fgX = pd.read_parquet(fg_feat)
    y_make = pd.read_parquet(fg_cur)["fg_made"].astype(int).values
    Xm = pd.DataFrame({c: df_fgX[c] if c in df_fgX.columns else 0 for c in make_bag["bags"][0].feat_cols})
    p_make = bag_mean_prob(make_bag, Xm)

    # --- Attempt ---
    att_feat = artifacts / "features_attempt.parquet"
    att_cur  = artifacts / "curated_attempt.parquet"
    if not att_feat.exists() or not att_cur.exists():
        raise FileNotFoundError("Need features_attempt.parquet and curated_attempt.parquet. Run 01 & 02.")

    with (artifacts / "attempt_bag.pkl").open("rb") as f:
        att_bag = pickle.load(f)
    df_attX = pd.read_parquet(att_feat)
    y_att = pd.read_parquet(att_cur)["attempt_fg"].astype(int).values
    Xa = pd.DataFrame({c: df_attX[c] if c in df_attX.columns else 0 for c in att_bag["bags"][0].feat_cols})
    p_att = bag_mean_prob(att_bag, Xa)

    # Metrics
    def safe_auc(y, p):
        try: return roc_auc_score(y, p)
        except Exception: return float("nan")

    make_json = {
        "rows": int(len(y_make)),
        "auc": float(safe_auc(y_make, p_make)),
        "brier": float(brier_score_loss(y_make, p_make)),
        "reliability": reliability(y_make, p_make, 10),
        "slice_60_68": reliability(y_make[(df_fgX["distance"]>=60)&(df_fgX["distance"]<=68)],
                                   p_make[(df_fgX["distance"]>=60)&(df_fgX["distance"]<=68)], 6)
    }
    att_json = {
        "rows": int(len(y_att)),
        "auc": float(safe_auc(y_att, p_att)),
        "brier": float(brier_score_loss(y_att, p_att)),
        "reliability": reliability(y_att, p_att, 10)
    }

    with (logs_dir / "make_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(make_json, f, indent=2)
    with (logs_dir / "attempt_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(att_json, f, indent=2)

    with (logs_dir / "calibration_report.txt").open("w", encoding="utf-8") as f:
        f.write("== Calibration Report ==\n\n")
        f.write(f"[MAKE] rows={make_json['rows']}, AUC={make_json['auc']:.4f}, Brier={make_json['brier']:.5f}\n")
        f.write("  Reliability bins (mean p vs observed):\n")
        for b in make_json["reliability"]["bins"]:
            f.write(f"    n={b['n']:5d}  p={b['p_mean']:.3f}  y={b['y_rate']:.3f}\n")
        f.write("\n  Long kicks 60–68:\n")
        for b in make_json["slice_60_68"]["bins"]:
            f.write(f"    n={b['n']:5d}  p={b['p_mean']:.3f}  y={b['y_rate']:.3f}\n")

        f.write("\n[ATTEMPT] rows={}, AUC={:.4f}, Brier={:.5f}\n".format(
            att_json["rows"], att_json["auc"], att_json["brier"]
        ))
        f.write("  Reliability bins:\n")
        for b in att_json["reliability"]["bins"]:
            f.write(f"    n={b['n']:5d}  p={b['p_mean']:.3f}  y={b['y_rate']:.3f}\n")

    logging.getLogger("08_calibration_eval").info("Wrote make/attempt calibration JSONs and report to %s", logs_dir)
    logging.getLogger("08_calibration_eval").info("Done.")

if __name__ == "__main__":
    main()
