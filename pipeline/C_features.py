#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_features.py — Physics- and context-aware feature engineering for FG make and attempt.

Inputs (from 01_ingest.py in cfg.output_dir):
  - curated_fg.parquet
  - curated_attempt.parquet

Optional (from cfg.assets_dir):
  - assets/stadiums.csv  (columns: team, stadium, altitude_m, uprights_azimuth_deg, roof, surface)

Outputs (to cfg.output_dir):
  - features_fg.parquet
  - features_attempt.parquet

Highlights:
  - Air density (altitude + temperature) as multiplicative factor.
  - Wind vector resolution: headwind / crosswind (m/s) using stadium uprights azimuth.
  - Distance tail ramps: (d-58)+, (d-60)+, (d-62)+ to ensure 62 ≠ 68 in downstream models.
  - Hash/angle extraction from description when present (best-effort).
  - Robustness: if stadiums.csv or wind direction missing, features fallback to NaN or neutral.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from A_config import (
    Config, setup_logging, ensure_dir,
    load_stadiums, f_to_kelvin, std_air_density_kg_m3,
    wind_components_mps,
)

# ----------------------------- Helpers ------------------------------------

# ---------- Stadiums: load + attach ----------
ROOF_INDOOR_KEYS = {"DOME", "FIXED", "DOME/FIXED", "INDOOR"}
ROOF_RETRACT_KEYS = {"RETRACTABLE", "RETRACTABLE ROOF"}

def load_stadiums_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    # Normalize columns we need
    for col in ["team", "team_fastr", "stadium_name", "altitude", "heading", "roof_type",
                "first_game_date", "last_game_date", "is_current", "tz", "lat", "lon"]:
        if col not in df.columns:
            df[col] = np.nan

    # Clean types
    df["team"] = df["team"].astype(str).str.strip().str.upper()
    df["team_fastr"] = df["team_fastr"].astype(str).str.strip().str.upper()
    df["stadium_name"] = df["stadium_name"].astype(str).str.strip()
    df["roof_type_norm"] = df["roof_type"].astype(str).str.strip().str.upper()
    df["is_current"] = df["is_current"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
    df["first_game_date"] = pd.to_datetime(df["first_game_date"], errors="coerce")
    df["last_game_date"] = pd.to_datetime(df["last_game_date"], errors="coerce")
    # altitude appears to be meters in your sample; keep as meters
    df["altitude_m"] = pd.to_numeric(df["altitude"], errors="coerce")
    df["stadium_heading_deg"] = pd.to_numeric(df["heading"], errors="coerce")  # 0-359 bearing of field axis

    # is_indoor from roof_type
    roof = df["roof_type_norm"].fillna("")
    is_dome = roof.isin(ROOF_INDOOR_KEYS)
    is_retract = roof.isin(ROOF_RETRACT_KEYS)
    # Treat retractable as outdoor by default (we don't know open/closed per game)
    df["is_indoor_stadium"] = np.where(is_dome, 1, 0).astype(int)
    df["is_roof_retractable"] = np.where(is_retract, 1, 0).astype(int)

    # Canonical team key: prefer team (ARI/ATL/…), else team_fastr
    df["team_key"] = df["team"].where(df["team"].notna() & df["team"].ne("nan"), df["team_fastr"])
    df["team_key"] = df["team_key"].astype(str).str.strip().str.upper()

    return df[[
        "team_key", "stadium_name", "altitude_m", "stadium_heading_deg",
        "is_indoor_stadium", "is_roof_retractable",
        "roof_type", "first_game_date", "last_game_date", "is_current", "lat", "lon", "tz"
    ]].copy()

def pick_stadium_row(stads: pd.DataFrame, team: str, game_date: pd.Timestamp) -> pd.Series | None:
    """Pick the stadium row for a team/date using interval match, then fallbacks."""
    if stads.empty:
        return None
    s = stads[stads["team_key"] == team]
    if s.empty:
        return None
    if pd.notna(game_date):
        in_window = s[
            (s["first_game_date"].notna()) & (s["last_game_date"].notna()) &
            (s["first_game_date"] <= game_date) & (game_date <= s["last_game_date"])
        ]
        if not in_window.empty:
            return in_window.sort_values("first_game_date").iloc[-1]
    # Fallback to "current"
    current = s[s["is_current"]].copy()
    if not current.empty:
        # If multiple currents (shared venues), just take the last row
        return current.iloc[-1]
    # Else the most recent by last_game_date or max first_game_date
    if s["last_game_date"].notna().any():
        return s.sort_values("last_game_date").iloc[-1]
    if s["first_game_date"].notna().any():
        return s.sort_values("first_game_date").iloc[-1]
    return s.iloc[-1]

def attach_stadium_features(df: pd.DataFrame,
                            stads: pd.DataFrame,
                            team_col_candidates=("home_team","posteam","pos_team"),
                            date_col="game_date") -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        out[date_col] = pd.NaT
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    # pick team column
    team_col = None
    for c in team_col_candidates:
        if c in out.columns:
            team_col = c
            break
    if team_col is None:
        # last-ditch: try 'team'
        team_col = "team"
        out[team_col] = out.get("posteam", "")
    out["team_key"] = out[team_col].astype(str).str.upper().str.strip()

    # Prepare destination columns
    for c in ["altitude_m","stadium_heading_deg","is_indoor_stadium","is_roof_retractable",
              "roof_type","stadium_name","stadium_tz","stadium_lat","stadium_lon"]:
        if c not in out.columns:
            out[c] = np.nan

    # Row-wise attach (vectorized merge is hard with intervals; this is fast enough for our sizes)
    rows = []
    for idx, r in out.iterrows():
        st = pick_stadium_row(stads, r["team_key"], r[date_col])
        if st is not None:
            rows.append((idx, st))
    if rows:
        for idx, st in rows:
            out.at[idx, "altitude_m"] = st["altitude_m"]
            out.at[idx, "stadium_heading_deg"] = st["stadium_heading_deg"]
            out.at[idx, "is_indoor_stadium"] = int(st["is_indoor_stadium"])
            out.at[idx, "is_roof_retractable"] = int(st["is_roof_retractable"])
            out.at[idx, "roof_type"] = st["roof_type"]
            out.at[idx, "stadium_name"] = st["stadium_name"]
            out.at[idx, "stadium_tz"] = st["tz"]
            out.at[idx, "stadium_lat"] = st["lat"]
            out.at[idx, "stadium_lon"] = st["lon"]

    # Improve 'is_indoor' used by downstream: prefer stadium flag when available
    if "is_indoor" not in out.columns:
        out["is_indoor"] = 0
    out["is_indoor"] = np.where(out["is_indoor_stadium"].notna(),
                                out["is_indoor_stadium"].astype("Int64").fillna(0),
                                out["is_indoor"]).astype(int)

    # Recompute wind components using stadium heading if we have wind_dir_deg + wind_mps
    # Conventions:
    #  - wind_dir_deg: meteorological "from" direction (0 = from north, clockwise)
    #  - stadium_heading_deg: bearing of one end zone -> the other along the long axis
    # We compute absolute components (magnitude) since offense direction flips.
    def recompute_wind(row):
        spd_mps = row.get("wind_mps", np.nan)
        wdir = row.get("wind_dir_deg", np.nan)
        head = row.get("stadium_heading_deg", np.nan)
        if not np.isfinite(spd_mps) or not np.isfinite(wdir) or not np.isfinite(head):
            return np.nan, np.nan
        # Convert meteo "from" to "to" by adding 180°
        to_deg = (wdir + 180.0) % 360.0
        # Relative to field axis
        rel = np.deg2rad((to_deg - head + 360.0) % 360.0)
        head_mps = abs(spd_mps * np.cos(rel))
        cross_mps = abs(spd_mps * np.sin(rel))
        return head_mps, cross_mps

    if "wind_mps" not in out.columns and "wind_mph" in out.columns:
        out["wind_mps"] = pd.to_numeric(out["wind_mph"], errors="coerce") * 0.44704

    if "wind_dir_deg" in out.columns:
        vals = out.apply(recompute_wind, axis=1, result_type="expand")
        out["wind_head_mps_precise"] = vals[0]
        out["wind_cross_mps_precise"] = vals[1]
        # Fill canonical columns if they exist, else create them
        for src, dst in [("wind_head_mps_precise","wind_head_mps"),
                         ("wind_cross_mps_precise","wind_cross_mps")]:
            if dst not in out.columns:
                out[dst] = np.nan
            out[dst] = out[dst].fillna(out[src])

    # Air density ratio using altitude (simple scale-height model; temp effects already modeled elsewhere)
    # rho ~ rho0 * exp(-h/H), H ≈ 8434 m
    H = 8434.0
    if "air_density_ratio" not in out.columns:
        out["air_density_ratio"] = 1.0
    out["air_density_ratio"] = np.where(out["altitude_m"].notna(),
                                        np.exp(-out["altitude_m"].astype(float) / H),
                                        out["air_density_ratio"])
    return out

_HASH_PATTERNS = [
    (re.compile(r"\b(left hash)\b", re.I), "L"),
    (re.compile(r"\b(right hash)\b", re.I), "R"),
    (re.compile(r"\b(middle|mid hash|center hash)\b", re.I), "M"),
]

def infer_hash_from_desc(desc: object) -> str:
    if not isinstance(desc, str):
        return "U"
    for pat, val in _HASH_PATTERNS:
        if pat.search(desc):
            return val
    return "U"  # Unknown

def tail_ramp(x: pd.Series, k: int) -> pd.Series:
    return np.clip(x.astype(float) - float(k), 0.0, None)

def add_common_env_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure these columns exist and are numeric:
      temp_F, wind_mph, wind_dir_deg, is_indoor
    """
    out = df.copy()
    for c in ["temp_F", "wind_mph", "wind_dir_deg", "is_indoor"]:
        if c not in out.columns:
            out[c] = np.nan
    out["temp_F"] = pd.to_numeric(out["temp_F"], errors="coerce")
    out["wind_mph"] = pd.to_numeric(out["wind_mph"], errors="coerce")
    out["wind_dir_deg"] = pd.to_numeric(out["wind_dir_deg"], errors="coerce")
    out["is_indoor"] = pd.to_numeric(out["is_indoor"], errors="coerce").fillna(0).astype(int)
    return out

def merge_stadium_features(df: pd.DataFrame, stadiums: pd.DataFrame) -> pd.DataFrame:
    """
    Join by home team if available; otherwise leave stadium fields NaN.
    """
    out = df.copy()
    if stadiums.empty:
        out["altitude_m"] = np.nan
        out["uprights_azimuth_deg"] = np.nan
        out["surface"] = np.nan
        out["roof_stadium"] = np.nan
        return out

    # Normalize columns
    st = stadiums.copy()
    st_cols = {c.lower(): c for c in st.columns}
    col_team = st_cols.get("team", None)
    col_alt  = st_cols.get("altitude_m", None)
    col_az   = st_cols.get("uprights_azimuth_deg", None)
    col_roof = st_cols.get("roof", None)
    col_surf = st_cols.get("surface", None)
    if not col_team:
        # Can't join — return with NaNs
        out["altitude_m"] = np.nan
        out["uprights_azimuth_deg"] = np.nan
        out["surface"] = np.nan
        out["roof_stadium"] = np.nan
        return out

    # Choose team column from df (home_team preferred)
    team_col = "home_team" if "home_team" in out.columns else ("posteam" if "posteam" in out.columns else None)
    if team_col is None:
        out["altitude_m"] = np.nan
        out["uprights_azimuth_deg"] = np.nan
        out["surface"] = np.nan
        out["roof_stadium"] = np.nan
        return out

    st_small = st[[col_team] + [x for x in [col_alt, col_az, col_roof, col_surf] if x]].copy()
    st_small = st_small.rename(columns={
        col_team: "home_team",
        **({col_alt: "altitude_m"} if col_alt else {}),
        **({col_az: "uprights_azimuth_deg"} if col_az else {}),
        **({col_roof: "roof_stadium"} if col_roof else {}),
        **({col_surf: "surface"} if col_surf else {}),
    })
    out = out.merge(st_small, how="left", on="home_team")
    return out

def compute_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute air density ratio and wind head/cross components relative to uprights.
    """
    out = df.copy()
    # Air density
    temp_K = f_to_kelvin(out["temp_F"].fillna(60.0))
    alt_m  = pd.to_numeric(out.get("altitude_m", np.nan), errors="coerce").fillna(0.0)
    rho    = [std_air_density_kg_m3(a, t) for a, t in zip(alt_m.values, temp_K.values)]
    rho0   = std_air_density_kg_m3(0.0, 288.15)  # ~1.225 kg/m^3 at 15C
    out["air_density_ratio"] = np.array(rho, dtype=float) / rho0

    # Wind components
    uprights_deg = pd.to_numeric(out.get("uprights_azimuth_deg", np.nan), errors="coerce")
    wmph = out["wind_mph"]
    wdir = out["wind_dir_deg"]
    head_list, cross_list = [], []
    for s, d, a in zip(wmph.values, wdir.values, uprights_deg.values):
        h, c = wind_components_mps(s, d, a) if np.isfinite(a) else (float("nan"), float("nan"))
        head_list.append(h)
        cross_list.append(c)
    out["wind_head_mps"] = head_list
    out["wind_cross_mps"] = cross_list

    return out

def add_distance_tails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distance tail ramps to let models distinguish 62 vs 68 yards clearly.
    """
    out = df.copy()
    out["d"] = pd.to_numeric(out["distance"], errors="coerce")
    for k in (58, 60, 62):
        out[f"d_tail_{k}"] = tail_ramp(out["d"], k)
    # Convert to meters for any physics downstream (not strictly required for models)
    out["distance_m"] = out["d"] * 0.9144
    return out

def add_hash_and_footedness(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    desc_col = "desc" if "desc" in out.columns else ("description" if "description" in out.columns else None)
    out["hash_lr"] = out[desc_col].apply(infer_hash_from_desc) if desc_col else "U"
    # Footedness placeholder (default R); can be merged later from an asset
    out["kicker_foot"] = "R"
    return out


# ----------------------------- Main ---------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Feature engineering for FG make and attempt models.")
    ap.add_argument("--output-dir")
    ap.add_argument("--assets-dir", default=".\\assets")
    ap.add_argument("--stadiums-csv", default=r"C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp\team_stadiums.csv")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = Config.from_args(output_dir=args.output_dir, log_level=args.log_level)
    setup_logging(cfg.log_level)
    log = logging.getLogger("features")

    artifacts = Path(cfg.output_dir)
    ensure_dir(artifacts)

    # Load curated parquet from B_ingest
    fg_cur = artifacts / "curated_fg.parquet"
    att_cur = artifacts / "curated_attempt.parquet"
    if not fg_cur.exists() or not att_cur.exists():
        raise FileNotFoundError("Run B_ingest.py first to produce curated_fg/attempt.")

    df_fg = pd.read_parquet(fg_cur)
    df_att = pd.read_parquet(att_cur)

    # ... your existing feature engineering (temp_F, wind_mph, wind_dir_deg, wind_*_mps, etc.) ...

    # ---- NEW: enrich with stadiums ----
    stad_fp = Path(args.stadiums_csv)
    if stad_fp.exists():
        stad_df = load_stadiums_csv(stad_fp)
        df_fg = attach_stadium_features(df_fg, stad_df, team_col_candidates=("home_team","posteam"))
        df_att = attach_stadium_features(df_att, stad_df, team_col_candidates=("home_team","posteam"))
        log.info("Attached stadium features (altitude/roof/heading) to FG and attempt frames.")
    else:
        log.warning("Stadiums CSV missing at %s; skipping stadium enrichment.", stad_fp)

    # Save to artifacts for downstream
    out_fg = artifacts / "features_fg.parquet"
    out_att = artifacts / "features_attempt.parquet"
    df_fg.to_parquet(out_fg, index=False)
    df_att.to_parquet(out_att, index=False)
    log.info("Wrote %s (%d rows) and %s (%d rows)", out_fg, len(df_fg), out_att, len(df_att))


if __name__ == "__main__":
    main()
