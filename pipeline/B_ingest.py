#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_ingest.py — Read raw pbp (parquet) and depth charts (CSV), curate FG + 4th-down attempt rows,
               and (optionally) emit a team→starter kicker roster JSON for the UI.

Outputs (to cfg.output_dir):
  - curated_fg.parquet
  - curated_attempt.parquet

Also writes (to cfg.ui_dir):
  - nfl_kicker_roster.json    # {team: {"starter_id": "...", "starter_name": "...", "as_of_season": 2024}}

Notes:
- If 2025 depth chart is missing, we fall back to 2024 for roster/starter info.
- Robust to varied schema names in pbp and depth charts.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from A_config import (
    Config, setup_logging,
    extract_temp_f, extract_wind_mph, extract_wind_dir_deg,
    ensure_dir,
)


# ----------------------------- Column helpers -----------------------------

def choose(cols: List[str], *candidates: str) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# ----------------------------- PBP ingestion -----------------------------

def read_pbp_parquets(pbp_root: Path, years: List[int]) -> pd.DataFrame:
    log = logging.getLogger("ingest.pbp")
    dfs: List[pd.DataFrame] = []
    for y in years:
        fp = Path(pbp_root) / f"play_by_play_{y}.parquet"
        if not fp.exists():
            log.warning("Missing PBP parquet: %s (skipping year %s)", fp, y)
            continue
        df = pd.read_parquet(fp)
        df["season_year"] = y
        dfs.append(df)
        log.info("Loaded %s (%s rows)", fp, f"{len(df):,}")
    if not dfs:
        raise FileNotFoundError("No PBP parquet files found for requested years.")
    out = pd.concat(dfs, ignore_index=True)
    log.info("Combined PBP rows: %s", f"{len(out):,}")
    return out


def curate_fg(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Extract FG attempts with outcome and key contextuals.
    """
    log = logging.getLogger("ingest.curate_fg")
    cols = list(df_all.columns)

    play_type = choose(cols, "play_type", "playtype", "play_type_nfl")
    fg_result = choose(cols, "field_goal_result", "fg_result", "fieldgoalresult")
    fg_attempt = choose(cols, "field_goal_attempt", "fg_attempt")
    kick_dist = choose(cols, "kick_distance", "field_goal_distance", "kickdistance")
    roof = choose(cols, "roof")
    weather = choose(cols, "weather", "weather_detail", "weather_description")
    desc = choose(cols, "desc", "description", "play_description", "play_desc")
    game_id = choose(cols, "game_id", "gameid")
    game_date = choose(cols, "game_date", "gamedate", "game_day")
    kicker_id = choose(cols, "kicker_player_id", "kicker_id", "kickerid")
    kicker_nm = choose(cols, "kicker_player_name", "kicker", "kicker_name")
    posteam = choose(cols, "posteam", "pos_team", "offense_team")
    home_team = choose(cols, "home_team")

    # Identify FG rows
    if play_type is not None:
        mask = df_all[play_type].astype(str).str.lower().eq("field_goal")
    elif fg_attempt is not None:
        mask = pd.to_numeric(df_all[fg_attempt], errors="coerce").fillna(0).astype(int).eq(1)
    elif fg_result is not None:
        mask = df_all[fg_result].notna()
    else:
        mask = pd.Series(False, index=df_all.index)

    fg = df_all.loc[mask].copy()

    # Outcome label
    if fg_result is not None:
        fg["fg_made"] = fg[fg_result].astype(str).str.lower().isin(["made", "good", "success"]).astype(int)
    elif desc is not None:
        fg["fg_made"] = fg[desc].astype(str).str.contains(r"\b(is )?good\b|\bmade\b", case=False, regex=True).astype(int)
    else:
        fg["fg_made"] = np.nan

    # Distance
    if kick_dist is not None:
        fg["distance"] = pd.to_numeric(fg[kick_dist], errors="coerce")
    elif desc is not None:
        fg["distance"] = pd.to_numeric(
            fg[desc].astype(str).str.extract(r"(\d+)\s*-\s*yard\s*field\s*goal", expand=False),
            errors="coerce"
        )
    else:
        fg["distance"] = np.nan

    # Environment
    if roof is not None:
        fg["roof_raw"] = fg[roof].astype(str).str.lower()
        fg["is_indoor"] = fg["roof_raw"].ne("outdoors").astype(int)
    else:
        fg["roof_raw"] = np.nan
        fg["is_indoor"] = 0

    # Weather
    if weather is not None:
        w = fg[weather].astype(str)
        fg["temp_F"] = w.apply(extract_temp_f)
        fg["wind_mph"] = w.apply(extract_wind_mph)
        fg["wind_dir_deg"] = w.apply(extract_wind_dir_deg)
    else:
        fg["temp_F"] = np.nan
        fg["wind_mph"] = np.nan
        fg["wind_dir_deg"] = np.nan

    # IDs / meta
    fg["game_id"] = fg[game_id] if game_id else np.arange(len(fg))
    fg["kicker_id"] = fg[kicker_id].astype(str) if kicker_id else "unknown"
    fg["kicker_name"] = fg[kicker_nm] if kicker_nm else fg["kicker_id"]
    fg["posteam"] = fg[posteam].astype(str) if posteam else "UNK"
    fg["home_team"] = fg[home_team].astype(str) if home_team else np.nan

    if game_date:
        fg["game_date"] = pd.to_datetime(df_all.loc[fg.index, game_date], errors="coerce")
    else:
        # Fallback: season mid-date
        fg["game_date"] = pd.to_datetime(fg["season_year"].astype(str) + "-10-15", errors="coerce")

    # Clean
    fg = fg.dropna(subset=["distance"]).copy()
    fg = fg[fg["fg_made"].isin([0, 1])]
    fg["distance"] = fg["distance"].astype(int).clip(10, 80)
    fg["temp_F"] = pd.to_numeric(fg["temp_F"], errors="coerce")
    fg["wind_mph"] = pd.to_numeric(fg["wind_mph"], errors="coerce")
    fg["wind_dir_deg"] = pd.to_numeric(fg["wind_dir_deg"], errors="coerce")
    fg["is_indoor"] = fg["is_indoor"].fillna(0).astype(int)

    # Impute missing weather mildly
    fg["temp_F"] = fg["temp_F"].fillna(fg["temp_F"].median())
    fg["wind_mph"] = fg["wind_mph"].fillna(fg["wind_mph"].median())

    # Restrict to plausible FG distances for modeling (18–68 for UI)
    fg = fg[(fg["distance"] >= 18) & (fg["distance"] <= 68)].reset_index(drop=True)

    log.info("Curated FG rows: %s", f"{len(fg):,}")
    return fg


def curate_attempt(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 4th-down decision rows where a FG attempt is an option and build the attempt label.
    """
    log = logging.getLogger("ingest.curate_attempt")
    cols = list(df_all.columns)

    play_type  = choose(cols, "play_type", "playtype", "play_type_nfl")
    down       = choose(cols, "down")
    ytg        = choose(cols, "ydstogo", "yards_to_go", "yds_to_go")
    yardline   = choose(cols, "yardline_100", "yardline")
    roof       = choose(cols, "roof")
    weather    = choose(cols, "weather", "weather_detail", "weather_description")
    game_id    = choose(cols, "game_id", "gameid")
    game_date  = choose(cols, "game_date", "gamedate", "game_day")
    score_diff = choose(cols, "score_differential", "score_diff")
    posteam    = choose(cols, "posteam", "pos_team", "offense_team")
    kicker_id  = choose(cols, "kicker_player_id", "kicker_id", "kickerid")
    home_team  = choose(cols, "home_team")
    half_sec   = choose(cols, "half_seconds_remaining", "half_seconds")

    if down is None or yardline is None or play_type is None or posteam is None:
        raise ValueError("Attempt curation requires down, yardline_100, play_type, posteam columns.")

    att = df_all.copy()

    # 4th down only
    att = att[pd.to_numeric(att[down], errors="coerce") == 4].copy()

    # FG distance estimate = yardline_100 + 17
    att["distance"] = pd.to_numeric(att[yardline], errors="coerce") + 17
    att = att.dropna(subset=["distance"]).copy()
    att["distance"] = att["distance"].astype(int)
    att = att[(att["distance"] >= 18) & (att["distance"] <= 68)].copy()

    # Attempt label
    att["attempt_fg"] = att[play_type].astype(str).str.lower().eq("field_goal").astype(int)

    # Env
    if roof is not None:
        att["roof_raw"] = att[roof].astype(str).str.lower()
        att["is_indoor"] = att["roof_raw"].ne("outdoors").astype(int)
    else:
        att["roof_raw"] = np.nan
        att["is_indoor"] = 0

    # Weather
    if weather is not None:
        w = att[weather].astype(str)
        att["temp_F"] = w.apply(extract_temp_f)
        att["wind_mph"] = w.apply(extract_wind_mph)
        att["wind_dir_deg"] = w.apply(extract_wind_dir_deg)
    else:
        att["temp_F"] = np.nan
        att["wind_mph"] = np.nan
        att["wind_dir_deg"] = np.nan

    att["temp_F"] = pd.to_numeric(att["temp_F"], errors="coerce")
    att["wind_mph"] = pd.to_numeric(att["wind_mph"], errors="coerce")
    att["wind_dir_deg"] = pd.to_numeric(att["wind_dir_deg"], errors="coerce")

    # Context
    att["ydstogo"] = pd.to_numeric(att[ytg], errors="coerce").fillna(0) if ytg else 0
    att["score_diff"] = pd.to_numeric(att[score_diff], errors="coerce").fillna(0) if score_diff else 0
    att["score_diff"] = att["score_diff"].clip(-21, 21)
    att["half_sec"] = pd.to_numeric(att[half_sec], errors="coerce").fillna(900) if half_sec else 900

    # IDs/dates/teams
    att["posteam"] = att[posteam].astype(str)
    att["home_team"] = att[home_team].astype(str) if home_team else np.nan
    att["game_id"] = att[game_id] if game_id else np.arange(len(att))
    if game_date:
        att["game_date"] = pd.to_datetime(att[game_date], errors="coerce")
    else:
        # Use season midpoint if not present
        year_col = choose(cols, "season", "season_year")
        if year_col:
            att["game_date"] = pd.to_datetime(att[year_col].astype(str) + "-10-15", errors="coerce")
        else:
            att["game_date"] = pd.NaT

    # Infer per-game kicker (mode) if available
    if kicker_id:
        kick_rows = df_all.loc[df_all[kicker_id].notna(), [game_id or "game_id", posteam or "posteam", kicker_id]].copy()
        kick_rows.columns = ["game_id", "posteam", "kicker_id"]
        mode_map = (
            kick_rows.groupby(["game_id", "posteam"])["kicker_id"]
            .agg(lambda s: s.astype(str).mode().iloc[0] if len(s) else np.nan)
            .to_dict()
        )
        att["kicker_id_game"] = [mode_map.get((g, t), np.nan) for g, t in zip(att["game_id"], att["posteam"])]
    else:
        att["kicker_id_game"] = np.nan

    att = att.reset_index(drop=True)
    log.info("Curated attempt rows: %s", f"{len(att):,}")
    return att


# ----------------------------- Depth charts → starter roster -----------------------------

def read_depth_charts(depth_root: Path, years: List[int]) -> pd.DataFrame:
    """
    Expect files like depth_charts_2024.csv; 2025 may be missing (that's fine).
    """
    log = logging.getLogger("ingest.depth")
    dfs: List[pd.DataFrame] = []
    for y in years:
        fp = Path(depth_root) / f"depth_charts_{y}.csv"
        if not fp.exists():
            log.warning("Missing depth chart: %s", fp)
            continue
        df = pd.read_csv(fp)
        df["season_year"] = y
        dfs.append(df)
        log.info("Loaded %s (%s rows)", fp, f"{len(df):,}")
    if not dfs:
        log.warning("No depth charts loaded at all; roster/starter info will be minimal.")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# --- NEW helper: fallback roster from curated FG ---
def roster_from_curated_fg(curated_fg_fp: Path, prefer_year: int = 2024) -> Dict[str, Dict[str, str]]:
    log = logging.getLogger("ingest.roster.fallback")
    if not curated_fg_fp.exists():
        log.warning("curated_fg.parquet not found at %s; returning empty roster.", curated_fg_fp)
        return {}

    df = pd.read_parquet(curated_fg_fp)
    # Prefer chosen year; else use most recent
    if "season_year" in df.columns and df["season_year"].notna().any():
        years = sorted(df["season_year"].dropna().unique())
        use_year = prefer_year if prefer_year in years else years[-1]
        df = df[df["season_year"] == use_year].copy()
    else:
        use_year = None

    # Need team, kicker_id, kicker_name
    for c in ["posteam", "kicker_id", "kicker_name"]:
        if c not in df.columns:
            log.warning("curated_fg missing '%s'; returning empty roster.", c)
            return {}

    # Use the mode kicker per team in that season
    df = df.dropna(subset=["posteam", "kicker_id"])
    if df.empty:
        return {}

    mode_kicker = (
        df.groupby("posteam")["kicker_id"]
        .agg(lambda s: s.astype(str).mode().iloc[0] if len(s) else "")
    )

    # Latest known name for that (team, kicker_id)
    if "game_date" in df.columns:
        df = df.sort_values(["posteam", "kicker_id", "game_date"])
    name_map = df.groupby(["posteam", "kicker_id"])["kicker_name"].last()

    roster: Dict[str, Dict[str, str]] = {}
    for team, kid in mode_kicker.items():
        kid = str(kid)
        kname = str(name_map.get((team, kid), kid))
        roster[str(team)] = {
            "starter_id": kid,
            "starter_name": kname,
            "as_of_season": str(use_year) if use_year is not None else ""
        }
    log.info("Fallback roster (from curated FG) built for %d teams (season=%s).", len(roster), use_year)
    return roster


# --- REPLACE your build_kicker_roster with this robust version ---
def build_kicker_roster(depth_df: pd.DataFrame, prefer_year: int, curated_fg_fp: Path) -> Dict[str, Dict[str, str]]:
    """
    Derive team -> starter kicker. If depth charts don't yield any K rows,
    fall back to curated FG to pick a starter per team.
    """
    log = logging.getLogger("ingest.roster")
    if depth_df is None or depth_df.empty:
        log.warning("Depth charts empty; falling back to curated FG.")
        return roster_from_curated_fg(curated_fg_fp, prefer_year=prefer_year)

    cols = list(depth_df.columns)
    team_col        = choose(cols, "team", "posteam", "club_code", "team_abbr")
    pos_col         = choose(cols, "position", "pos", "depth_chart_position", "position_group")
    player_id_col   = choose(cols, "player_id", "gsis_id", "nfl_id", "pfr_id")
    player_name_col = choose(cols, "player_name", "name", "full_name", "player")

    if team_col is None or pos_col is None:
        log.warning("Could not locate team/position columns in depth charts; falling back to curated FG.")
        return roster_from_curated_fg(curated_fg_fp, prefer_year=prefer_year)

    df = depth_df.copy()
    # Normalize and filter positions
    df[pos_col] = df[pos_col].astype(str).str.upper().str.strip()
    K_POS = {"K", "PK", "KICKER"}
    df_k = df[df[pos_col].isin(K_POS)].copy()

    if df_k.empty:
        log.warning("No K/PK/KICKER rows found in depth charts; falling back to curated FG.")
        return roster_from_curated_fg(curated_fg_fp, prefer_year=prefer_year)

    # Seasons present after filter
    if "season_year" not in df_k.columns or df_k["season_year"].dropna().empty:
        log.warning("No season_year after K filter; falling back to curated FG.")
        return roster_from_curated_fg(curated_fg_fp, prefer_year=prefer_year)

    seasons_available = sorted(df_k["season_year"].dropna().unique())
    use_year = prefer_year if prefer_year in seasons_available else seasons_available[-1]
    d = df_k[df_k["season_year"] == use_year].copy()

    # Starter = smallest depth/rank; if no numeric depth, first per team
    depth_col = choose(cols, "depth", "depth_chart_order", "rank", "dc_rank", "order", "depth_team")
    if depth_col and depth_col in d.columns:
        d["_depth"] = pd.to_numeric(d[depth_col], errors="coerce")
        d = d.sort_values([team_col, "_depth"], na_position="last")
        starters = d.groupby(team_col, as_index=False).first()
    else:
        starters = d.sort_values([team_col]).drop_duplicates(subset=[team_col], keep="first")

    roster: Dict[str, Dict[str, str]] = {}
    for _, r in starters.iterrows():
        team = str(r[team_col])
        pid  = str(r[player_id_col]) if (player_id_col and player_id_col in starters.columns and pd.notna(r[player_id_col])) else ""
        pname= str(r[player_name_col]) if (player_name_col and player_name_col in starters.columns and pd.notna(r[player_name_col])) else ""
        # If ID+name both missing, try fg fallback for that single team
        if not pid and not pname:
            fg_fallback = roster_from_curated_fg(curated_fg_fp, prefer_year=use_year)
            entry = fg_fallback.get(team, {})
            pid = entry.get("starter_id", "")
            pname = entry.get("starter_name", "")
        if team and (pid or pname):
            roster[team] = {"starter_id": pid, "starter_name": pname, "as_of_season": str(use_year)}

    log.info("Built roster for %d teams from depth charts (season %s).", len(roster), use_year)
    return roster


# ----------------------------- Main ---------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ingest PBP + depth charts; output curated FG/Attempt and roster JSON.")
    ap.add_argument("--pbp-root")
    ap.add_argument("--depth-root")
    ap.add_argument("--output-dir")
    ap.add_argument("--ui-dir")
    ap.add_argument("--years", help='e.g. "2020:2025" (inclusive)')
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = Config.from_args(
        pbp_root=args.pbp_root,
        depth_root=args.depth_root,
        output_dir=args.output_dir,
        ui_dir=args.ui_dir,
        years=args.years,
        log_level=args.log_level,
    )
    setup_logging(cfg.log_level)
    log = logging.getLogger("ingest")

    log.info("Config:\n%s", cfg.to_json())

    # 1) Load all PBP
    df_all = read_pbp_parquets(cfg.pbp_root, cfg.years)

    # 2) Curate FG + Attempt frames
    fg = curate_fg(df_all)
    att = curate_attempt(df_all)

    # 3) Save curated parquet
    ensure_dir(cfg.output_dir)
    fg_out = cfg.output_dir / "curated_fg.parquet"
    att_out = cfg.output_dir / "curated_attempt.parquet"
    fg.to_parquet(fg_out, index=False)
    att.to_parquet(att_out, index=False)
    log.info("Wrote %s (%s rows)", fg_out, f"{len(fg):,}")
    log.info("Wrote %s (%s rows)", att_out, f"{len(att):,}")

    # 4) Depth charts → roster JSON (optional)
    depth_df = read_depth_charts(cfg.depth_root, cfg.years)
    roster = build_kicker_roster(depth_df, prefer_year=2024, curated_fg_fp=fg_out)
    if roster:
        ui_json = Path(cfg.ui_dir) / "nfl_kicker_roster.json"
        with ui_json.open("w", encoding="utf-8") as f:
            import json
            json.dump(roster, f, indent=2)
        log.info("Wrote roster JSON: %s", ui_json)
    else:
        log.warning("No roster JSON written (missing/empty depth charts).")

    log.info("Done.")


if __name__ == "__main__":
    main()
