#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D_make_physics_gam.py — Physics-ish GAM baseline for league FG make probability.

- Smooth function of distance (cubic splines) + linear env/physics terms
- Out-of-fold isotonic calibration for honest probabilities
- Saves a portable model: artifacts/make_gam.pkl
- Writes a small report: logs/make_gam_report.txt

Inputs:
  artifacts/curated_fg.parquet
  artifacts/features_fg.parquet  (from C_features.py)

Outputs:
  artifacts/make_gam.pkl
  logs/make_gam_report.txt
"""

from __future__ import annotations
import logging, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from A_config import Config, setup_logging, ensure_dir

DIST_MIN, DIST_MAX = 18, 68

class GamIsoModel:
    """Spline distance + linear env (LogReg) with isotonic calibration on model scores."""
    def __init__(self, pipe: Pipeline, iso: IsotonicRegression, feat_cols: list[str]):
        self.pipe = pipe
        self.iso = iso
        self.feat_cols = feat_cols

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for c in self.feat_cols:
            if c not in X: X[c] = 0
        return X[self.feat_cols].copy()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._X(df)
        raw = self.pipe.predict_proba(X)[:, 1]
        cal = np.clip(self.iso.predict(raw), 1e-9, 1-1e-9)
        return np.column_stack([1 - cal, cal])

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Required/used features (robust to missing; we fill with 0 where needed)
    want = [
        "distance",
        "is_indoor",
        "temp_F",
        "wind_mph",          # kept for compatibility
        "wind_head_mps",
        "wind_cross_mps",
        "air_density_ratio",
        "altitude_m"
    ]
    out = df.copy()
    for c in want:
        if c not in out.columns: out[c] = 0
    # Standard cleaning
    out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
    out = out[(out["distance"] >= DIST_MIN) & (out["distance"] <= DIST_MAX)].copy()
    out["is_indoor"] = pd.to_numeric(out["is_indoor"], errors="coerce").fillna(0).astype(int)
    for c in ["temp_F","wind_mph","wind_head_mps","wind_cross_mps","air_density_ratio","altitude_m"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    # centers
    out["temp_c"] = (out["temp_F"] - 60.0) / 20.0
    out["density_c"] = out["air_density_ratio"] - 1.0
    return out

def fit_gam(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> GamIsoModel:
    # Spline on distance; linear env terms
    feat_cols = ["distance","is_indoor","temp_c","wind_head_mps","wind_cross_mps","density_c","altitude_m"]
    distance_spline = Pipeline([
        ("spline", SplineTransformer(degree=3, n_knots=7, extrapolation="linear")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    coltx = ColumnTransformer([
        ("dist_spline", distance_spline, ["distance"]),
        ("env_passthrough", "passthrough", ["is_indoor","temp_c","wind_head_mps","wind_cross_mps","density_c","altitude_m"])
    ])
    pipe = Pipeline([
        ("ct", coltx),
        ("clf", LogisticRegression(max_iter=2000, C=3.0, solver="lbfgs"))
    ])

    # OOF for isotonic
    gkf = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups)))))
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        pipe.fit(X.iloc[tr][feat_cols], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te][feat_cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)

    # Refit on all
    pipe.fit(X[feat_cols], y)
    return GamIsoModel(pipe, iso, feat_cols)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Physics-ish GAM baseline for FG make.")
    ap.add_argument("--output-dir")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = Config.from_args(output_dir=args.output_dir, log_level=args.log_level)
    setup_logging(cfg.log_level)
    log = logging.getLogger("make_gam")

    artifacts = Path(cfg.output_dir)
    logs = ensure_dir(artifacts.parent / "logs")
    fg_feat = artifacts / "features_fg.parquet"
    fg_cur  = artifacts / "curated_fg.parquet"
    if not fg_feat.exists() or not fg_cur.exists():
        raise FileNotFoundError("Run B_ingest.py and C_features.py first.")

    Xdf = pd.read_parquet(fg_feat)
    y = pd.read_parquet(fg_cur)["fg_made"].astype(int).values
    if "game_id" in Xdf.columns:
        groups = Xdf["game_id"].astype(str).values
    else:
        groups = np.arange(len(Xdf)).astype(str)

    X = select_columns(Xdf)
    y = y[X.index.values]

    model = fit_gam(X, y, groups[X.index.values])

    # Quick metrics
    p = model.predict_proba(X)[:, 1]
    def safe_auc(yy, pp):
        from sklearn.metrics import roc_auc_score
        try: return roc_auc_score(yy, pp)
        except Exception: return float("nan")
    with (logs / "make_gam_report.txt").open("w", encoding="utf-8") as f:
        f.write("== Make GAM Report ==\n")
        f.write(f"Rows: {len(y):,}\n")
        f.write(f"AUC:   {safe_auc(y, p):.4f}\n")
        f.write(f"Brier: {brier_score_loss(y, p):.5f}\n")

        # Sanity: neutral outdoor 55 vs 60 — build explicit 1-row frames (no .update)
        X55 = pd.DataFrame({
            "distance": [55],
            "is_indoor": [0],
            "temp_c": [0.0],
            "wind_head_mps": [0.0],
            "wind_cross_mps": [0.0],
            "density_c": [0.0],
            "altitude_m": [0.0],
        })
        X60 = X55.copy()
        X60.loc[:, "distance"] = 60

        p55 = model.predict_proba(X55)[0, 1].item()
        p60 = model.predict_proba(X60)[0, 1].item()
        f.write(f"Neutral check: p55={p55:.3f} p60={p60:.3f}  (expect p60<p55: {p60<=p55})\n")


    # Save baseline
    with (artifacts / "make_gam.pkl").open("wb") as f:
        pickle.dump(model, f)

    log.info("Saved artifacts/make_gam.pkl and logs/make_gam_report.txt")
    log.info("Done.")

if __name__ == "__main__":
    main()
