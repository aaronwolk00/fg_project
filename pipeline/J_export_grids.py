#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
J_export_grids.py — Consolidated UI export + manifest.

Re-exports:
  - ui/fg_prob_grid_distance_env_temp_wind.json
  - ui/fg_attempt_grid_distance_env_context.json
  - ui/kicker_deltas_logit_banded.json          (rebuilt if parquet available)
  - ui/kicker_deltas_by_distance.json           (rebuilt if parquet available)
  - ui/attempt_sensitivity.json                 (if present)

Also writes:
  - ui/manifest.json

Inputs:
  artifacts/make_bag.pkl
  artifacts/attempt_bag.pkl
  artifacts/kicker_deltas_by_distance.parquet (optional)
"""

from __future__ import annotations
import argparse, json, logging, pickle, hashlib, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from A_config import Config, setup_logging, ensure_dir

# ---------- ADD: class needed to unpickle models ----------
from sklearn.isotonic import IsotonicRegression  # noqa: F401
class IsoWrapper:
    """Matches the class used when pickling the bags in E/G steps."""
    def __init__(self, base_model, iso: IsotonicRegression, feat_cols: List[str]):
        self.base = base_model
        self.iso = iso
        self.feat_cols = feat_cols
    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        for c in self.feat_cols:
            if c not in Xc:
                Xc[c] = 0
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce").fillna(0)
        return Xc[self.feat_cols]
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = self._X(X)
        raw = self.base.predict_proba(Xc)[:, 1]
        cal = np.clip(self.iso.predict(raw), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - cal, cal])

# ----------------- constants -----------------
DIST_MIN, DIST_MAX = 18, 68
DISTANCES = list(range(DIST_MIN, DIST_MAX + 1))
TEMP_GRID = list(range(30, 100, 5))
WIND_GRID = list(range(0, 22, 2))
ENV_OPTS  = ["indoor","outdoor"]

SCORE_BINS = ["trail", "close", "lead"]
TIME_BINS  = ["early", "mid", "late"]
YTG_BINS   = ["short", "med", "long"]
REP_SCORE  = {"trail": -7, "close": 0, "lead": +7}
REP_TIME   = {"early": 1200, "mid": 600, "late": 120}
REP_YTG    = {"short": 2, "med": 5, "long": 9}

# --------------- utils ---------------
def bag_mean_prob(bag: Dict, X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(X)[:,1] for m in bag["bags"]]
    return np.mean(np.vstack(preds), axis=0)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

# --------------- export funcs ---------------
def export_make_grid(make_bag: Dict, ui_dir: Path) -> Path:
    rows = []
    for env in ENV_OPTS:
        is_indoor = 1 if env == "indoor" else 0
        for temp in TEMP_GRID:
            for wind in WIND_GRID:
                wind_mps = 0.44704 * wind if is_indoor == 0 else 0.0
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
                P = bag_mean_prob(make_bag, Xg)
                for d, pm in zip(DISTANCES, P):
                    rows.append({
                        "distance": int(d), "indoor_outdoor": env, "temp_F": int(temp),
                        "wind_mph": int(wind if is_indoor==0 else 0),
                        "prob_mean": round(float(pm), 6)
                    })
    payload = {
        "meta": {
            "note": "Exported by J_export_grids from artifacts/make_bag.pkl",
            "distances": [DIST_MIN, DIST_MAX],
            "temps": TEMP_GRID, "winds": WIND_GRID, "env": ENV_OPTS
        },
        "grid": rows
    }
    out = ui_dir / "fg_prob_grid_distance_env_temp_wind.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out

def export_attempt_grid(att_bag: Dict, ui_dir: Path) -> Path:
    rows = []
    for env in ["indoor","outdoor"]:
        is_indoor = 1 if env == "indoor" else 0
        for score in SCORE_BINS:
            for t in TIME_BINS:
                for ytg in YTG_BINS:
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
                    P = bag_mean_prob(att_bag, Xg)
                    for d, p in zip(DISTANCES, P):
                        rows.append({
                            "distance": int(d), "indoor_outdoor": env,
                            "score_bin": score, "time_bin": t, "ytg_bin": ytg,
                            "prob_attempt_mean": round(float(p), 6)
                        })
    payload = {
        "meta": {
            "note": "Exported by J_export_grids from artifacts/attempt_bag.pkl",
            "distances": [DIST_MIN, DIST_MAX],
            "env": ["indoor","outdoor"], "score_bins": SCORE_BINS,
            "time_bins": TIME_BINS, "ytg_bins": YTG_BINS,
            "rep_values": {"score": REP_SCORE, "time_half_sec": REP_TIME, "ydstogo": REP_YTG}
        },
        "grid": rows
    }
    out = ui_dir / "fg_attempt_grid_distance_env_context.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out

def rebuild_kicker_jsons_from_parquet(artifacts: Path, ui_dir: Path) -> List[Path]:
    out_paths: List[Path] = []
    fp = artifacts / "kicker_deltas_by_distance.parquet"
    if not fp.exists():
        return out_paths
    df = pd.read_parquet(fp)
    # by-distance
    payload = []
    for kid, g in df.groupby("kicker_id"):
        kname = str(g["kicker_name"].iloc[0])
        by_d = {
            int(r.distance): {
                "delta_logit": float(r.delta_logit),
                "se": float(r.se_logit),
                "attempts_eff": float(r.attempts_eff)
            } for r in g.itertuples(index=False)
        }
        payload.append({"kicker_id": str(kid), "kicker_name": kname, "by_distance": by_d})
    out_by = ui_dir / "kicker_deltas_by_distance.json"
    out_by.write_text(json.dumps(payload), encoding="utf-8")
    out_paths.append(out_by)

    # banded
    def band_of(d: int) -> str:
        if d <= 39:  return "short"
        if d <= 49:  return "mid"
        if d <= 59:  return "long"
        return "xlong"
    df["band"] = df["distance"].astype(int).apply(band_of)
    band_payload = []
    for kid, g in df.groupby("kicker_id"):
        kname = str(g["kicker_name"].iloc[0])
        bands = {}
        for band, gb in g.groupby("band"):
            w = gb["attempts_eff"].replace(0, 1.0)
            delta = np.average(gb["delta_logit"], weights=w)
            se = np.sqrt(np.average(gb["se_logit"]**2, weights=w))
            bands[band] = {"delta_logit": float(delta), "se": float(se), "n_eff": float(w.sum())}
        band_payload.append({"kicker_id": str(kid), "kicker_name": kname,
                             "attempts_eff": float(g["attempts_eff"].sum()), "bands": bands})
    out_band = ui_dir / "kicker_deltas_logit_banded.json"
    out_band.write_text(json.dumps(band_payload, indent=2), encoding="utf-8")
    out_paths.append(out_band)
    return out_paths

def write_manifest(ui_dir: Path, files: List[str]) -> Path:
    items = []
    for rel in files:
        p = ui_dir / rel
        if not p.exists(): continue
        stat = p.stat()
        items.append({
            "file": rel,
            "bytes": stat.st_size,
            "mtime_epoch": int(stat.st_mtime),
            "sha256": sha256_file(p)
        })
    manifest = {"generated_at_epoch": int(time.time()), "items": items}
    out = ui_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out

def main():
    ap = argparse.ArgumentParser(description="Re-export UI grids and write manifest.")
    ap.add_argument("--output-dir")
    ap.add_argument("--ui-dir")
    ap.add_argument("--rebuild", choices=["all","make","attempt","kicker","manifest"], default="all")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = Config.from_args(output_dir=args.output_dir, ui_dir=args.ui_dir, log_level=args.log_level)
    setup_logging(cfg.log_level)
    log = logging.getLogger("export")

    artifacts = Path(cfg.output_dir)
    ui_dir = Path(cfg.ui_dir)
    ensure_dir(ui_dir)

    exported = []

    # Make grid
    if args.rebuild in ("all","make"):
        mkp = artifacts / "make_bag.pkl"
        if not mkp.exists():
            log.error("Missing %s — run D and E first.", mkp)
        else:
            with mkp.open("rb") as f:
                make_bag = pickle.load(f)
            out = export_make_grid(make_bag, ui_dir)
            log.info("Wrote %s", out)
            exported.append(out.name)

    # Attempt grid
    if args.rebuild in ("all","attempt"):
        abp = artifacts / "attempt_bag.pkl"
        if not abp.exists():
            log.error("Missing %s — run G first.", abp)
        else:
            with abp.open("rb") as f:
                att_bag = pickle.load(f)
            out = export_attempt_grid(att_bag, ui_dir)
            log.info("Wrote %s", out)
            exported.append(out.name)

    # Kicker JSONs (optional rebuild)
    if args.rebuild in ("all","kicker"):
        need = [ui_dir/"kicker_deltas_logit_banded.json", ui_dir/"kicker_deltas_by_distance.json"]
        if not all(p.exists() for p in need):
            made = rebuild_kicker_jsons_from_parquet(artifacts, ui_dir)
            for p in made:
                log.info("Rebuilt %s", p)
                exported.append(p.name)

    # Include attempt sensitivity if present
    sens = ui_dir / "attempt_sensitivity.json"
    if sens.exists(): exported.append(sens.name)

    # Manifest
    if args.rebuild in ("all","manifest"):
        files = [
            "fg_prob_grid_distance_env_temp_wind.json",
            "fg_attempt_grid_distance_env_context.json",
            "kicker_deltas_logit_banded.json",
            "kicker_deltas_by_distance.json",
            "attempt_sensitivity.json"
        ]
        man = write_manifest(ui_dir, files)
        log.info("Wrote %s", man)

    log.info("Export finished.")

if __name__ == "__main__":
    main()
