#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_coupling_and_wp.py — Learn coupling between Attempt% and Kicker Make Advantage.

Inputs:
  - artifacts/attempt_bag.pkl
  - artifacts/features_attempt.parquet
  - artifacts/features_fg.parquet            (fallback make model training)
  - ui/kicker_deltas_by_distance.json        (preferred)
    or ui/kicker_deltas_logit_banded.json    (fallback)

Optional preferred (if earlier steps saved it):
  - artifacts/make_bag.pkl                   (monotone LGBM + isotonic for league make)

Output:
  - ui/attempt_sensitivity.json
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from A_config import Config, setup_logging

# --------------------- Helpers ---------------------

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

def fit_make_fallback(features_fg_fp: Path):
    """Train a quick monotone LGBM + isotonic using features_fg.parquet (league make)."""
    log = logging.getLogger("coupling.make_fallback")
    df = pd.read_parquet(features_fg_fp)
    y = pd.read_parquet(features_fg_fp.parent / "curated_fg.parquet")["fg_made"].astype(int).values

    feat_cols = ["distance","is_indoor","temp_F","wind_mph","air_density_ratio","wind_head_mps","wind_cross_mps"]
    for c in feat_cols:
        if c not in df.columns: df[c] = 0
        df["is_indoor"] = pd.to_numeric(df["is_indoor"], errors="coerce").fillna(0).astype(int)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # isotonic via OOF
    groups = df.get("game_id", pd.Series(np.arange(len(df)))).astype(str).values
    gkf = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups)))))
    oof = np.zeros(len(y))
    for tr, te in gkf.split(df, y, groups):
        base = lgb.LGBMClassifier(
            objective="binary", learning_rate=0.05, n_estimators=900,
            num_leaves=31, max_depth=6, min_data_in_leaf=60,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
            verbose=-1, max_bin=63, n_jobs=-1,
            monotone_constraints=[-1, 0, 0, 0, 0, 0, 0]  # distance monotone ↓
        )
        base.fit(df.iloc[tr][feat_cols], y[tr])
        oof[te] = base.predict_proba(df.iloc[te][feat_cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)

    base_full = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, n_estimators=900,
        num_leaves=31, max_depth=6, min_data_in_leaf=60,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
        verbose=-1, max_bin=63, n_jobs=-1,
        monotone_constraints=[-1, 0, 0, 0, 0, 0, 0]
    )
    base_full.fit(df[feat_cols], y)
    log.info("Trained fallback make model (monotone LGBM + isotonic).")
    return {"bags": [IsoWrapper(base_full, iso, feat_cols)], "feat_cols": feat_cols}

def bag_mean_prob(bag: Dict, X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(X)[:, 1] for m in bag["bags"]]
    return np.mean(np.vstack(preds), axis=0)

def load_kicker_delta_func(ui_dir: Path):
    """Return a function f(kicker_id, distance)->delta_logit using preferred file, else banded blender."""
    bydist = ui_dir / "kicker_deltas_by_distance.json"
    banded = ui_dir / "kicker_deltas_logit_banded.json"

    if bydist.exists():
        data = json.loads(bydist.read_text(encoding="utf-8"))
        lookup = {d["kicker_id"]: {int(k): v for k, v in d["by_distance"].items()} for d in data}
        def f(kid: str, d: int) -> float:
            m = lookup.get(str(kid), {})
            if d in m: return float(m[d]["delta_logit"])
            # nearest neighbor fallback
            if m:
                nd = min(m.keys(), key=lambda x: abs(x - d))
                return float(m[nd]["delta_logit"])
            return 0.0
        return f

    # Fallback to banded blender
    if banded.exists():
        data = json.loads(banded.read_text(encoding="utf-8"))
        lut = {d["kicker_id"]: d["bands"] for d in data}
        def f(kid: str, d: int) -> float:
            bnds = lut.get(str(kid), {})
            # blend across boundaries with 2-yard ramp
            BL, bounds = 2, {"short":39, "mid":49, "long":59}
            def weights(dd):
                if dd <= bounds["short"] - BL: return {"short":1.0}
                if dd <  bounds["short"] + BL:
                    t = (dd - (bounds["short"] - BL)) / (2*BL); return {"short":1-t,"mid":t}
                if dd <= bounds["mid"] - BL: return {"mid":1.0}
                if dd <  bounds["mid"] + BL:
                    t = (dd - (bounds["mid"] - BL)) / (2*BL); return {"mid":1-t,"long":t}
                if dd <= bounds["long"] - BL: return {"long":1.0}
                if dd <  bounds["long"] + BL:
                    t = (dd - (bounds["long"] - BL)) / (2*BL); return {"long":1-t,"xlong":t}
                return {"xlong":1.0}
            ww = weights(int(d))
            delta = 0.0
            for band, w in ww.items():
                if band in bnds:
                    delta += float(bnds[band].get("delta_logit", 0.0)) * w
            return float(delta)
        return f

    # Nothing available
    logging.getLogger("coupling").warning("No kicker deltas JSON found; using zero deltas.")
    return lambda kid, d: 0.0

# --------------------- Main ---------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Couple Attempt% with kicker make advantage.")
    ap.add_argument("--output-dir")
    ap.add_argument("--ui-dir")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = Config.from_args(output_dir=args.output_dir, ui_dir=args.ui_dir, log_level=args.log_level)
    setup_logging(cfg.log_level)
    log = logging.getLogger("07_coupling")

    artifacts = Path(cfg.output_dir)
    ui_dir = Path(cfg.ui_dir)

    # Load attempt bag
    att_bag_fp = artifacts / "attempt_bag.pkl"
    if not att_bag_fp.exists():
        raise FileNotFoundError(f"Missing {att_bag_fp}. Run 06_attempt_model.py first.")
    with att_bag_fp.open("rb") as f:
        att_bag = pickle.load(f)

    # Load attempt features
    att_feat = artifacts / "features_attempt.parquet"
    att_cur = artifacts / "curated_attempt.parquet"
    if not att_feat.exists() or not att_cur.exists():
        raise FileNotFoundError("Missing attempt features/curated parquet. Run 01_ingest & 02_features.")

    X_att = pd.read_parquet(att_feat)
    y_att = pd.read_parquet(att_cur)["attempt_fg"].astype(int).values

    # Load or build make bag
    make_bag_fp = artifacts / "make_bag.pkl"
    if make_bag_fp.exists():
        with make_bag_fp.open("rb") as f:
            make_bag = pickle.load(f)
        log.info("Loaded existing make_bag.pkl")
    else:
        make_bag = fit_make_fallback(artifacts / "features_fg.parquet")

    # Prepare inputs for coupling
    # League make prob for each attempt row: use env subset of features_fg
    make_feats = make_bag["bags"][0].feat_cols
    Xm = pd.DataFrame({c: X_att[c] if c in X_att.columns else 0 for c in make_feats})
    Pm = bag_mean_prob(make_bag, Xm)   # league make
    lPm = logit(Pm)

    # Base attempt prob from att bag
    att_feats = att_bag["bags"][0].feat_cols
    Xa = pd.DataFrame({c: X_att[c] if c in X_att.columns else 0 for c in att_feats})
    Pa_base = bag_mean_prob(att_bag, Xa)
    lPa_base = logit(Pa_base)

    # Kicker deltas
    kicker_ids = pd.read_parquet(att_cur).get("kicker_id_game", pd.Series(["__UNK__"]*len(X_att))).astype(str).tolist()
    distances = pd.to_numeric(X_att["distance"], errors="coerce").fillna(40).astype(int).tolist()
    f_delta = load_kicker_delta_func(ui_dir)
    deltas = np.array([f_delta(k, d) for k, d in zip(kicker_ids, distances)], dtype=float)

    # Adjusted make prob (for information only)
    lPm_kicker = lPm + deltas
    # delta_make_logit feature:
    delta_make_logit = deltas

    # Fit logistic regression: Attempt ~ A + C*logit(Pa_base) + B*delta_make_logit
    X_lr = np.column_stack([lPa_base, delta_make_logit])
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e6)  # effectively unregularized
    lr.fit(X_lr, y_att)
    A = float(lr.intercept_[0])
    C = float(lr.coef_[0, 0])
    B = float(lr.coef_[0, 1])

    # Export sensitivity
    sens = {
        "note": "logit(Attempt) = A + C*logit(P_attempt_base) + B*(kicker delta on logit(make))",
        "A": A, "B": B, "C": C,
        "meta": {"n_rows": int(len(y_att))}
    }
    out_json = ui_dir / "attempt_sensitivity.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(sens, f, indent=2)
    log.info("Wrote %s  (A=%.3f, B=%.3f, C=%.3f)", out_json, A, B, C)

    log.info("Done.")

if __name__ == "__main__":
    main()
