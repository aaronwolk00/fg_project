#!/usr/bin/env python3
"""
Builds per-venue logit adjustments from kick-level residuals.

Inputs (flexible):
- kicks_training.csv  (must include: distance, made(0/1), env[indoor/outdoor], temp_F, wind_mph, altitude_m, stadium_id)
- OR if stadium_id is missing, we infer from team via team_stadiums.csv (team -> stadium_id, env, altitude_m)

We compute, for each kick i:
  p_hat_i  = baseline league make% from your grid (approx via binning)
  w_i      = p_hat_i * (1 - p_hat_i)         (Fisher weight)
  delta_i  = (made_i - p_hat_i) / max(w_i,1e-6)   (approx. logit residual)
Venue delta = weighted mean(delta_i) with ridge shrinkage vs sample size.

Output:
venue_adjustments.json:
{
  "meta": { "schema": 1, "built_at": "...", "n_kicks": 12345, "ridge_n0": 120 },
  "venues": [
    {"stadium_id":"ARI_2024","delta_logit":0.18,"n":220,"se":0.23},
    ...
  ]
}
"""
import json, pathlib, math, pandas as pd, numpy as np
from datetime import datetime

KICKS_CSV = pathlib.Path("kicks_training.csv")      # adapt if needed
VENUES_CSV= pathlib.Path("team_stadiums.csv")       # optional fallback
GRID_JSON = pathlib.Path("fg_prob_grid_distance_env_temp_wind.json")  # league grid
OUT_JSON  = pathlib.Path("venue_adjustments.json")

def inv_logit(z): return 1 / (1 + np.exp(-z))
def clip01(p): return np.minimum(1-1e-6, np.maximum(1e-6, p))

def bin5(x):  return int(round(float(x)/5.0)*5)    # temp bins ~5F
def bin2(x):  return int(round(float(x)/2.0)*2)    # wind bins ~2mph
def bin50m(x):return int(round(float(x)/50.0)*50)  # altitude bins 50m
def rd_dist(x):return int(round(float(x)))

def grid_lookup(df, env, d, tF, w, altm):
    # nearest-cell match (mirrors UI)
    dR, tR, wR, aR = rd_dist(d), bin5(tF), bin2(w), bin50m(altm)
    sub = df[(df["env"]==env) & (df["distance"]==dR) & (df["temp_F"]==tR) & (df["wind_mph"]==wR) & (df["altitude_m"]==aR)]
    if len(sub): return float(sub["prob_mean"].iloc[0])
    # fallback: env+distance only
    sub = df[(df["env"]==env) & (df["distance"]==dR)]
    if len(sub): return float(sub["prob_mean"].iloc[0])
    return 0.5

def main():
    if not KICKS_CSV.exists():
        raise SystemExit(f"Missing {KICKS_CSV.resolve()}")
    if not GRID_JSON.exists():
        raise SystemExit(f"Missing {GRID_JSON.resolve()} (league grid)")

    kicks = pd.read_csv(KICKS_CSV)
    grid  = pd.read_json(GRID_JSON)

    # normalize grid rows
    g = pd.DataFrame(grid["grid"])
    g["env"]         = g["indoor_outdoor"].fillna(g.get("env")).fillna("indoor")
    g["distance"]    = g["distance"].astype(int)
    g["temp_F"]      = g.get("temp_F", 60).astype(int)
    g["wind_mph"]    = g.get("wind_mph", 0).astype(int)
    g["altitude_m"]  = g.get("altitude_m", 0).astype(int)
    g["prob_mean"]   = g["prob_mean"].astype(float)

    # ensure stadium_id
    if "stadium_id" not in kicks.columns and VENUES_CSV.exists():
        venues = pd.read_csv(VENUES_CSV)
        team_col = "team_fastr" if "team_fastr" in venues.columns else "team"
        kicks = kicks.merge(
            venues[[team_col, "stadium_id"]],
            left_on="posteam", right_on=team_col, how="left"
        )

    kicks = kicks.dropna(subset=["stadium_id"])
    if kicks.empty:
        raise SystemExit("No stadium_id available to compute venue deltas.")

    # required columns with safe defaults
    for c, default in [
        ("env","indoor"), ("distance",45), ("temp_F",60), ("wind_mph",0), ("altitude_m",0), ("made",0)
    ]:
        if c not in kicks.columns: kicks[c] = default

    # compute p_hat from grid (row-wise)
    p = []
    for r in kicks.itertuples(index=False):
        p.append(grid_lookup(g, getattr(r, "env"), getattr(r, "distance"),
                             getattr(r, "temp_F"), getattr(r, "wind_mph"), getattr(r, "altitude_m")))
    kicks["p_hat"] = clip01(np.array(p))
    w = kicks["p_hat"] * (1 - kicks["p_hat"])
    kicks["w"] = w.clip(1e-6)

    # per-kick approx logit residual needed to match the outcome
    # delta ~= (y - p) / (p(1-p))
    kicks["delta_logit"] = (kicks["made"] - kicks["p_hat"]) / kicks["w"]

    # aggregate by venue with ridge shrinkage
    ridge_n0 = 120  # pseudo-counts; tune if desired
    agg = kicks.groupby("stadium_id").agg(n=("delta_logit", "size"),
                                          wsum=("w", "sum"),
                                          delta=("delta_logit", "mean"),
                                          var=("delta_logit", "var")).reset_index()
    agg["var"] = agg["var"].fillna(0.0)
    # shrink toward 0 on logit scale
    shrink = agg["n"] / (agg["n"] + ridge_n0)
    agg["delta_logit_shrunk"] = agg["delta"] * shrink
    # SE for display (rough)
    agg["se"] = np.sqrt(agg["var"].clip(1e-9) / agg["n"].clip(1))

    out = {
      "meta": {
        "schema": 1,
        "built_at": datetime.utcnow().isoformat()+"Z",
        "n_kicks": int(len(kicks)),
        "ridge_n0": int(ridge_n0)
      },
      "venues": [
        {
          "stadium_id": str(r.stadium_id),
          "delta_logit": float(r.delta_logit_shrunk),
          "n": int(r.n),
          "se": float(r.se)
        } for r in agg.itertuples(index=False)
      ]
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON.resolve()} for {len(agg)} venues.")

if __name__ == "__main__":
    main()
