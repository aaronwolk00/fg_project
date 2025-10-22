#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_kicker_hier_bayes.py — Distance-smoothed, recency-weighted kicker make deltas.

Inputs (from 01_ingest / 02_features in cfg.output_dir):
  - curated_fg.parquet          (needs: distance, fg_made, game_date, kicker_id, kicker_name, season_year, posteam, home_team)
  - features_fg.parquet         (adds: temp_F, wind_mph, wind_dir_deg, air_density_ratio, wind_head_mps, wind_cross_mps, etc.)

Outputs:
  - ui/kicker_deltas_logit_banded.json   # legacy, for current UI (short/mid/long/xlong)
  - ui/kicker_deltas_by_distance.json    # v2, 18..68 per-kicker deltas
  - artifacts/kicker_deltas_by_distance.parquet
  - logs/kicker_hier_bayes_report.txt
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from A_config import Config, setup_logging, ensure_dir

# ---------------------------- Config knobs ----------------------------
DIST_MIN, DIST_MAX = 18, 68
DISTANCES = np.arange(DIST_MIN, DIST_MAX + 1)

# Recency / smoothing
RECENCY_HALF_LIFE_DAYS = 360.0
KERNEL_SIGMA = 2.0  # distance smoothing (yards)

# Prior strength by band
PRIOR_STRENGTH_BY_BAND = {
    "short": 150.0,  # <=39
    "mid":   120.0,  # 40-49
    "long":   80.0,  # 50-59
    "xlong":  20.0,  # >=60 (let data speak, but still stabilize)
}

# ---------------------------- Helpers ----------------------------

def dist_band(d: int) -> str:
    if d <= 39:  return "short"
    if d <= 49:  return "mid"
    if d <= 59:  return "long"
    return "xlong"

def logit(p: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def exp_recency_weight(age_days: float, half_life: float = RECENCY_HALF_LIFE_DAYS) -> float:
    return 0.5 ** (max(age_days, 0.0) / max(half_life, 1.0))

def gaussian_kernel(dx: np.ndarray, sigma: float = KERNEL_SIGMA) -> np.ndarray:
    return np.exp(-0.5 * (dx / sigma) ** 2)

@dataclass
class BandedPrior:
    p_league: float
    alpha0: float
    beta0: float

# ---------------------------- Core ----------------------------

def build_recency_weights(df: pd.DataFrame) -> pd.DataFrame:
    today = pd.to_datetime(df["game_date"].max())
    age_days = (today - df["game_date"]).dt.days.astype(float)
    w = age_days.apply(exp_recency_weight)
    return w

def league_curve(df: pd.DataFrame) -> Dict[int, float]:
    """Smoothed league make probability by distance d in DISTANCES."""
    # Effective weights = recency only. We'll smooth across distance via kernel around each d.
    df = df.copy()
    df["w_rec"] = build_recency_weights(df)
    out = {}
    for d in DISTANCES:
        k = gaussian_kernel(df["distance"].values - d, KERNEL_SIGMA)
        w = df["w_rec"].values * k
        n_eff = w.sum()
        s_eff = (w * df["fg_made"].values).sum()
        p = (s_eff / n_eff) if n_eff > 0 else 0.5
        out[int(d)] = float(p)
    return out

def prior_strength_for_distance(d: int) -> float:
    return PRIOR_STRENGTH_BY_BAND[dist_band(d)]

def kicker_curves(df: pd.DataFrame, league_p: Dict[int, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - df_delta_d: rows (kicker_id, kicker_name, distance, attempts_eff, p_post, delta_logit, se_logit)
      - df_banded:  aggregated legacy bands for UI
    """
    log = logging.getLogger("kicker")
    df = df.copy()
    df["w_rec"] = build_recency_weights(df)

    # Pre-group by kicker for efficiency
    curves = []
    band_rows = []

    for kid, g in df.groupby("kicker_id", sort=False):
        kname = str(g["kicker_name"].iloc[0]) if "kicker_name" in g.columns else kid
        # pre-cache arrays
        gd = g["distance"].values.astype(float)
        gm = g["fg_made"].values.astype(float)
        gw = g["w_rec"].values.astype(float)

        # Distance-level posteriors
        for d in DISTANCES:
            k = gaussian_kernel(gd - d, KERNEL_SIGMA)
            w = gw * k
            n_eff = float(w.sum())
            s_eff = float((w * gm).sum())

            pL = float(league_p[int(d)])
            strength = float(prior_strength_for_distance(int(d)))
            a0 = strength * pL
            b0 = strength * (1.0 - pL)

            # Posterior ~ Beta(a0 + s_eff, b0 + (n_eff - s_eff)) (treat s_eff as "soft successes")
            alpha = a0 + s_eff
            beta = b0 + max(n_eff - s_eff, 0.0)
            p_post = alpha / (alpha + beta)

            # delta on logit scale
            dlogit = logit(p_post) - logit(pL)

            # approximate se of logit via delta method
            var_p = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
            eps = 1e-9
            var_logit = var_p / (max(p_post, eps) ** 2 * max(1 - p_post, eps) ** 2)
            se_logit = float(np.sqrt(max(var_logit, 1e-12)))

            curves.append({
                "kicker_id": kid, "kicker_name": kname,
                "distance": int(d),
                "attempts_eff": n_eff,
                "p_post": float(p_post),
                "delta_logit": float(dlogit),
                "se_logit": se_logit,
            })

        # Legacy bands (aggregate with recency weights)
        g2 = g.copy()
        g2["band"] = g2["distance"].astype(int).apply(lambda x: dist_band(x))
        for band, gb in g2.groupby("band"):
            n_eff = float(gb["w_rec"].sum())
            s_eff = float((gb["w_rec"] * gb["fg_made"]).sum())
            # League prior by band's representative distance
            rep_d = {"short": 35, "mid": 45, "long": 55, "xlong": 63}[band]
            pL = league_p.get(rep_d, 0.5)
            strength = PRIOR_STRENGTH_BY_BAND[band]
            a0, b0 = strength * pL, strength * (1 - pL)
            alpha = a0 + s_eff
            beta = b0 + max(n_eff - s_eff, 0.0)
            p_post = alpha / (alpha + beta)
            dlogit = logit(p_post) - logit(pL)
            var_p = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
            eps = 1e-9
            var_logit = var_p / (max(p_post, eps) ** 2 * max(1 - p_post, eps) ** 2)
            se_logit = float(np.sqrt(max(var_logit, 1e-12)))
            band_rows.append({
                "kicker_id": kid, "kicker_name": kname,
                "band": band, "n_eff": n_eff,
                "delta_logit": float(dlogit), "se": se_logit
            })

    df_delta_d = pd.DataFrame(curves)
    df_banded = pd.DataFrame(band_rows)
    return df_delta_d, df_banded

# ---------------------------- IO / CLI ----------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build kicker deltas with distance smoothing + recency weighting.")
    ap.add_argument("--output-dir")
    ap.add_argument("--ui-dir")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = Config.from_args(output_dir=args.output_dir, ui_dir=args.ui_dir, log_level=args.log_level)
    setup_logging(cfg.log_level)
    log = logging.getLogger("05_kicker_hier_bayes")

    artifacts = Path(cfg.output_dir)
    ui_dir = Path(cfg.ui_dir)
    logs_dir = ensure_dir(Path(cfg.output_dir).parent / "logs")

    fg_cur = artifacts / "curated_fg.parquet"
    if not fg_cur.exists():
        raise FileNotFoundError(f"Missing {fg_cur}. Run 01_ingest.py first.")

    df_fg = pd.read_parquet(fg_cur)
    required = ["distance", "fg_made", "game_date", "kicker_id", "kicker_name"]
    for c in required:
        if c not in df_fg.columns:
            raise ValueError(f"curated_fg.parquet missing required column: {c}")

    df_fg = df_fg[(df_fg["distance"] >= DIST_MIN) & (df_fg["distance"] <= DIST_MAX)].copy()
    df_fg["game_date"] = pd.to_datetime(df_fg["game_date"], errors="coerce")

    # League curve by distance
    league_p = league_curve(df_fg)

    # Kicker curves
    df_delta_d, df_banded = kicker_curves(df_fg, league_p)

    # Persist per-distance parquet + JSON
    dist_parquet = artifacts / "kicker_deltas_by_distance.parquet"
    df_delta_d.to_parquet(dist_parquet, index=False)

    dist_json = ui_dir / "kicker_deltas_by_distance.json"
    # Compact JSON: one record per kicker with {distance: {delta_logit,se,attempts_eff}}
    payload = []
    for kid, g in df_delta_d.groupby("kicker_id"):
        kname = str(g["kicker_name"].iloc[0])
        by_d = {
            int(r.distance): {
                "delta_logit": float(r.delta_logit),
                "se": float(r.se_logit),
                "attempts_eff": float(r.attempts_eff)
            } for r in g.itertuples(index=False)
        }
        payload.append({"kicker_id": str(kid), "kicker_name": kname, "by_distance": by_d})
    with dist_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    log.info("Wrote %s and %s", dist_parquet, dist_json)

    # Legacy banded JSON (for existing UI)
    band_json = ui_dir / "kicker_deltas_logit_banded.json"
    band_payload = []
    grp = df_banded.groupby("kicker_id", sort=False)
    for kid, g in grp:
        kname = str(g["kicker_name"].iloc[0])
        bands = {}
        for _, r in g.iterrows():
            bands[r["band"]] = {"delta_logit": float(r["delta_logit"]), "se": float(r["se"]), "n_eff": float(r["n_eff"])}
        attempts_eff = float(df_delta_d[df_delta_d["kicker_id"] == kid]["attempts_eff"].sum())
        band_payload.append({"kicker_id": str(kid), "kicker_name": kname, "attempts_eff": attempts_eff, "bands": bands})
    with band_json.open("w", encoding="utf-8") as f:
        json.dump(band_payload, f, indent=2)
    log.info("Wrote %s", band_json)

    # Simple text report
    report = logs_dir / "kicker_hier_bayes_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("== Kicker Hierarchical (Empirical Bayes) Report ==\n")
        f.write(f"Kick distances: {DIST_MIN}..{DIST_MAX}\n")
        f.write(f"Kernel sigma: {KERNEL_SIGMA} yards; Recency half-life: {RECENCY_HALF_LIFE_DAYS} days\n")
        f.write("Prior strengths by band: " + str(PRIOR_STRENGTH_BY_BAND) + "\n\n")
        sample = df_delta_d.sample(min(len(df_delta_d), 20), random_state=42)
        for r in sample.itertuples():
            f.write(f"{r.kicker_name:20s} d={r.distance:2d} Δlogit={r.delta_logit:+.3f} se={r.se_logit:.3f}\n")
    log.info("Wrote %s", report)

    log.info("Done.")

if __name__ == "__main__":
    main()
