#!/usr/bin/env python3
"""
Export UI JSONs from trained models + kicker deltas.

Inputs:
  --make-model      artifacts/make_bag.pkl
  --attempt-model   artifacts/attempt_bag.pkl
  --kicker-deltas   artifacts/kicker_deltas_by_distance.parquet
Outputs (to --out-dir):
  - fg_prob_grid_distance_env_temp_wind.json
  - fg_attempt_grid_distance_env_context.json
  - kicker_deltas_by_distance.json
  - manifest.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent

def load_make_model(path):   return joblib.load(path)
def load_attempt_model(path):return joblib.load(path)

def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def export_kicker_deltas(parquet_path: str, out_json: Path):
    df = pd.read_parquet(parquet_path)
    # keep only {kicker_id, kicker_name, by_distance}
    rows = []
    for r in df.itertuples(index=False):
        rows.append({
            "kicker_id": str(getattr(r, "kicker_id")),
            "kicker_name": str(getattr(r, "kicker_name", "")),
            "by_distance": dict(getattr(r, "by_distance"))
        })
    out = {"meta": {"schema": 1}, "kickers": rows}
    save_json(out, out_json)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-model", required=True)
    ap.add_argument("--attempt-model", required=True)
    ap.add_argument("--kicker-deltas", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- (A) export league make% grid ----------
    make_model = load_make_model(args.make_model)
    distances = np.arange(18, 69)
    temps = np.arange(0, 111, 5)
    winds = np.arange(0, 26, 2)
    envs = ["indoor", "outdoor"]
    rows = []
    for env in envs:
        for d in distances:
            for t in temps:
                for w in winds:
                    p = float(make_model.predict_proba([[env, d, t, w]])[0]) \
                        if hasattr(make_model, "predict_proba") else float(make_model.predict([[env, d, t, w]])[0])
                    rows.append({
                        "env": env, "distance": int(d),
                        "temp_F": int(t), "wind_mph": int(w),
                        "prob_mean": p
                    })
    make_json = {"meta":{"schema_version":2}, "grid": rows}
    save_json(make_json, out_dir / "fg_prob_grid_distance_env_temp_wind.json")

    # ---------- (B) export attempt% grid ----------
    attempt_model = load_attempt_model(args.attempt_model)
    score_bins = ["trail","close","lead"]
    time_bins  = ["early","mid","late"]
    ytg_bins   = ["short","med","long"]
    rows = []
    for env in envs:
        for d in distances:
            for s in score_bins:
                for t in time_bins:
                    for y in ytg_bins:
                        p = float(attempt_model.predict_proba([[env, d, s, t, y]])[0]) \
                            if hasattr(attempt_model, "predict_proba") else float(attempt_model.predict([[env, d, s, t, y]])[0])
                        rows.append({
                            "env": env, "distance": int(d),
                            "score_bin": s, "time_bin": t, "ytg_bin": y,
                            "prob_attempt_mean": p
                        })
    attempt_json = {"meta":{"schema_version":2}, "grid": rows}
    save_json(attempt_json, out_dir / "fg_attempt_grid_distance_env_context.json")

    # ---------- (C) kicker deltas ----------
    export_kicker_deltas(args.kicker_deltas, out_dir / "kicker_deltas_by_distance.json")

    # ---------- (D) manifest ----------
    manifest = {
        "generated_at_epoch": int(datetime.utcnow().timestamp()),
        "model_hashes": {
            "make": Path(args.make_model).name,
            "attempt": Path(args.attempt_model).name,
            "kicker_deltas": Path(args.kicker_deltas).name,
        }
    }
    save_json(manifest, out_dir / "manifest.json")

    print(f"UI JSONs written to: {out_dir}")

if __name__ == "__main__":
    main()
