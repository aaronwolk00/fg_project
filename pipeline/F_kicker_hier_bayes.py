#!/usr/bin/env python3
"""
Hierarchical Bayesian kicker model (PyMC / PyTensor) — robust I/O.

Changes vs prior:
- Infers 'made' from many schemas (field_goal_result, fg_result, is_good, etc.).
- CLI overrides: --made-col, --kicker-id-col, --env-col.
- Still exports per-kicker logit deltas by distance (18..68) and an ArviZ .nc.

Example:
  python F_kicker_hier_bayes.py ^
    --data .\artifacts\features_fg.parquet ^
    --made-col field_goal_result ^
    --kicker-id-col kicker_id ^
    --num-samples 600 --num-tune 600 --target-accept 0.9 ^
    --out .\artifacts\kicker_hier_bayes.nc ^
    --export .\artifacts\kicker_deltas_by_distance.parquet
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# -------------------- Defaults --------------------
BASE_DIR = Path(__file__).resolve().parent
ART      = BASE_DIR / "artifacts"
DEFAULT_DATA_CANDIDATES = [
    ART / "features_fg.parquet",
    ART / "curated_fg.parquet",
    BASE_DIR / "kicks_training.csv",
]

# Accept many label columns (strings or 0/1)
MADE_SYNONYMS = [
    "made", "is_made", "fg_made", "made_fg", "is_good", "good", "fg_good",
    "kick_made", "kick_good", "y", "label"
]
MADE_STRING_RESULT_COLS = [
    "field_goal_result", "fg_result", "result", "kick_result", "outcome"
]  # expect strings like "made/good/missed/blocked/no good"

KICKER_ID_SYNONYMS = [
    "kicker_id","player_id","gsis_id","nfl_id","pfr_player_id","id","player"
]
ENV_SYNONYMS = ["env","indoor_outdoor","environment","roof_env"]

# -------------------- Helpers --------------------

def _auto_data() -> str | None:
    for p in DEFAULT_DATA_CANDIDATES:
        if p.exists():
            return str(p)
    return None

def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower()==".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower()==".csv":
        return pd.read_csv(p)
    raise SystemExit(f"Unsupported data file: {p}")

def _coerce_bool01(s: pd.Series) -> pd.Series:
    """Coerce a column to 0/1 (bool/int). Works for bools, 0/1 ints, 'True'/'False', etc."""
    x = s.copy()
    if x.dtype == bool:
        return x.astype(int)
    if pd.api.types.is_numeric_dtype(x):
        return (x.astype(float) > 0).astype(int)
    # string-like
    y = x.astype(str).str.strip().str.lower()
    return y.isin(["1","true","t","y","yes","made","good","success"]).astype(int)

def _infer_made(df: pd.DataFrame, override: str|None) -> pd.Series:
    if override and override in df.columns:
        col = df[override]
        if pd.api.types.is_string_dtype(col):
            # map strings
            y = col.astype(str).str.strip().str.lower()
            return y.isin(["made","good","success"]).astype(int)
        return _coerce_bool01(col)

    # hard boolean/01 columns
    for c in MADE_SYNONYMS:
        if c in df.columns:
            return _coerce_bool01(df[c])

    # string result columns
    for c in MADE_STRING_RESULT_COLS:
        if c in df.columns:
            y = df[c].astype(str).str.strip().str.lower()
            return y.isin(["made","good","g","successful","success"]).astype(int)

    raise SystemExit(
        "Could not infer 'made'. Supply --made-col <colname> or add one of: "
        f"{', '.join(MADE_SYNONYMS + MADE_STRING_RESULT_COLS)}"
    )

def _infer_kicker_id(df: pd.DataFrame, override: str|None) -> pd.Series:
    if override and override in df.columns:
        return df[override].astype(str)
    for c in KICKER_ID_SYNONYMS:
        if c in df.columns:
            return df[c].astype(str)
    raise SystemExit("Could not infer a kicker id column. Use --kicker-id-col or add one of: "
                     + ", ".join(KICKER_ID_SYNONYMS))

def _infer_env(df: pd.DataFrame, override: str|None) -> pd.Series:
    if override and override in df.columns:
        s = df[override]
    else:
        s = None
        for c in ENV_SYNONYMS:
            if c in df.columns:
                s = df[c]; break
        if s is None:
            return pd.Series(["indoor"]*len(df), index=df.index)
    s = s.astype(str).str.lower().str.strip()
    return np.where(s.isin(["outdoor","o","open"]), "outdoor", "indoor")

def _prepare_df(path: str, made_col: str|None, kid_col: str|None, env_col: str|None) -> pd.DataFrame:
    df = _load_df(path).copy()

    # Outcome
    df["made"] = _infer_made(df, made_col)

    # IDs & names
    df["kicker_id"] = _infer_kicker_id(df, kid_col)
    if "kicker_name" not in df.columns:
        for c in ["player_name","name","full_name","player"]:
            if c in df.columns:
                df["kicker_name"] = df[c]; break
        if "kicker_name" not in df.columns:
            df["kicker_name"] = df["kicker_id"]

    # Env + numerics
    df["env"] = _infer_env(df, env_col)

    for c, dflt in {"distance":45, "temp_F":60, "wind_mph":0, "altitude_m":0}.items():
        if c not in df.columns:
            df[c] = dflt
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(dflt)

    if "obs_weight" not in df.columns:
        df["obs_weight"] = 1.0

    # reasonable ranges
    df = df[(df["distance"]>=18) & (df["distance"]<=68)].copy()
    return df

# -------------------- Features --------------------

def make_rbf_basis(x: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    x = x[:, None]
    return np.exp(-0.5 * ((x - centers[None,:]) / width)**2)

def build_design(df: pd.DataFrame):
    x_d = df["distance"].to_numpy(float)
    xc  = x_d.mean()
    xz_d = (x_d - xc) / 10.0

    centers = np.arange(20, 69, 8)  # 20..68 step 8
    width = 6.0
    Phi = make_rbf_basis(x_d, centers, width)

    temp = (df["temp_F"].to_numpy(float) - 60.0)/20.0
    wind = df["wind_mph"].to_numpy(float)/10.0
    alt  = df["altitude_m"].to_numpy(float)/1000.0
    env  = (df["env"].astype(str).str.lower()=="outdoor").to_numpy(int)

    X = np.column_stack([Phi, temp, wind, alt, env])
    colnames = [f"rbf_{i}" for i in range(Phi.shape[1])] + ["temp","wind","alt","outdoor"]

    kicker_ids, inv = np.unique(df["kicker_id"].astype(str), return_inverse=True)
    k_idx = inv.astype(int)

    y = df["made"].astype(int).to_numpy()
    w = df["obs_weight"].astype(float).to_numpy()

    meta = {
        "centers": centers, "width": width, "xc": xc,
        "design_cols": colnames,
        "kicker_ids": kicker_ids,
        "kicker_names": df.groupby("kicker_id")["kicker_name"].first().reindex(kicker_ids).to_numpy(object),
    }
    return X, xz_d, y, w, k_idx, meta

# -------------------- Model --------------------

def build_model(X, xz_d, y, k_idx, min_kicks=15):
    n, p = X.shape
    K = k_idx.max() + 1
    _, counts = np.unique(k_idx, return_counts=True)
    prior_scale = np.where(counts >= min_kicks, 1.0, 0.3).astype("float32")

    with pm.Model() as m:
        beta = pm.Normal("beta", sigma=1.0, shape=p)
        intercept = pm.Normal("intercept", sigma=2.0)

        sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.5)
        a_raw = pm.Normal("a_raw", 0.0, 1.0, shape=K)
        b_raw = pm.Normal("b_raw", 0.0, 1.0, shape=K)

        # NOTE: prior_scale is per-kicker; broadcast via index at likelihood time
        a_k = a_raw * sigma_a
        b_k = b_raw * sigma_b

        eta = intercept + pt.dot(pt.as_tensor_variable(X), beta) \
              + a_k[k_idx] + b_k[k_idx] * pt.as_tensor_variable(xz_d) * prior_scale[k_idx]

        pm.Bernoulli("y", logit_p=eta, observed=y)

    return m

def sample_posterior(model, num_samples=800, num_tune=800, target_accept=0.9, seed=42):
    with model:
        idata = pm.sample(
            draws=num_samples, tune=num_tune, target_accept=target_accept,
            chains=2, random_seed=seed, progressbar=True
        )
    return idata

# -------------------- Posterior utilities --------------------

def posterior_deltas_by_distance(idata, meta, distances=np.arange(18,69)):
    beta = idata.posterior["beta"].stack(draw=("chain","draw")).values     # [p,S]
    intercept = idata.posterior["intercept"].stack(draw=("chain","draw")).values  # [S]
    sigma_a = idata.posterior["sigma_a"].stack(draw=("chain","draw")).values
    sigma_b = idata.posterior["sigma_b"].stack(draw=("chain","draw")).values
    a_raw   = idata.posterior["a_raw"].stack(draw=("chain","draw")).values  # [K,S]
    b_raw   = idata.posterior["b_raw"].stack(draw=("chain","draw")).values  # [K,S]

    centers = meta["centers"]; width = meta["width"]; xc = meta["xc"]
    def rbf_grid(d): return np.exp(-0.5 * ((d - centers)/width)**2)

    rows=[]
    K = a_raw.shape[0]
    for k in range(K):
        a_k = a_raw[k] * sigma_a
        b_k = b_raw[k] * sigma_b
        grid={}
        for d in distances:
            Phi = rbf_grid(d)
            X_fix = np.concatenate([Phi, [0,0,0,0]])       # neutral context
            eta_L = intercept + beta.T @ X_fix             # [S]
            dz = (d - xc)/10.0
            eta_K = eta_L + a_k + b_k * dz
            delta = eta_K - eta_L
            grid[int(d)] = {"delta_logit": float(delta.mean()),
                            "se": float(delta.std(ddof=1))}
        rows.append({
            "kicker_id": str(meta["kicker_ids"][k]),
            "kicker_name": str(meta["kicker_names"][k]),
            "by_distance": grid
        })
    return pd.DataFrame(rows)

# (compat) optional
def posterior_predict(*args, **kwargs):  # keeps J or other imports happy
    raise NotImplementedError("posterior_predict is not used by the exporter.")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--made-col", default=None, help="Override column to derive made (0/1 or string)")
    ap.add_argument("--kicker-id-col", default=None, help="Override kicker id column")
    ap.add_argument("--env-col", default=None, help="Override environment column")
    ap.add_argument("--min-kicks", type=int, default=15)
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--num-tune", type=int, default=800)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ART / "kicker_hier_bayes.nc"))
    ap.add_argument("--export", default=str(ART / "kicker_deltas_by_distance.parquet"))
    args = ap.parse_args()

    data_path = args.data or _auto_data()
    if not data_path:
        print("No training data found. Looked for:\n  --data <your file>\n  " +
              "\n  ".join(str(p.relative_to(BASE_DIR)) for p in DEFAULT_DATA_CANDIDATES))
        sys.exit(1)

    df = _prepare_df(data_path, args.made_col, args.kicker_id_col, args.env_col)

    X, xz, y, w, k_idx, meta = build_design(df)
    model = build_model(X, xz, y, k_idx, min_kicks=args.min_kicks)
    idata = sample_posterior(model, args.num_samples, args.num_tune, args.target_accept, args.seed)

    out_nc = Path(args.out); out_nc.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, out_nc)

    df_out = posterior_deltas_by_distance(idata, meta)
    out_pq = Path(args.export); out_pq.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_pq, index=False)

    print(f"Wrote {out_pq} and {out_nc}")

if __name__ == "__main__":
    main()
