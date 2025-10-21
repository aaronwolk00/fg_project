#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FG Pipeline — Professional Upgrade (SECTION 1/2)

This module focuses on **data ingestion, cleaning, feature building, and roster creation**.
SECTION 2 will cover **model training, calibration, ensembling, coupling, grid exports**, and reports.

Key upgrades in this section:
- Robust file handling for **parquet** (preferred) with **CSV fallback** (e.g., sample uploads).
- Integrates **depth charts (2024)** to build a team→starter kicker roster; for seasons without
  depth charts (e.g., 2025), falls back to **pbp-derived mode kicker** per team/week.
- Strong schema normalization with tolerant column pickers, resilient weather parsers, and
  clean indoor/outdoor logic.
- Clear validation checks, logging, and deterministic behavior.
- Persists curated datasets that SECTION 2 will train on.

Outputs written to: <OUTPUT_DIR>
  - curated_fg.parquet                   # clean FG attempts w/ features & labels
  - curated_attempt.parquet              # 4th-down decision rows w/ attempt label
  - nfl_kicker_roster.json               # team list w/ starter and status list
  - data_report.txt                      # summary stats & validation messages

Usage (examples):
  python fg_pipeline_section1.py \
      --pbp-root "C:/Users/awolk/Documents/NFELO/Other Data/nflverse/pbp" \
      --depth-root "C:/Users/awolk/Documents/NFELO/Other Data/nflverse/depth_charts" \
      --years 2020:2025

  # With the uploaded samples in this workspace as fallback:
  python fg_pipeline_section1.py --use-samples

Notes:
- Keep paths Windows-escaped or use forward slashes. Both are supported by pathlib.
- SECTION 2 expects these curated files in OUTPUT_DIR.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------- Configuration -----------------------------

@dataclass
class Config:
    pbp_root: Path
    depth_root: Path
    output_dir: Path
    years: List[int] = field(default_factory=lambda: list(range(2020, 2026)))
    use_samples: bool = False  # if True, will try /mnt/data sample csvs

    def ensure(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def default_config(args: Optional[argparse.Namespace] = None) -> Config:
    # Defaults derived from user's description
    pbp_default = Path(r"C:/Users/awolk/Documents/NFELO/Other Data/nflverse/pbp")
    depth_default = Path(r"C:/Users/awolk/Documents/NFELO/Other Data/nflverse/depth_charts")
    out_default = pbp_default / "model_outputs2"

    cfg = Config(
        pbp_root=Path(getattr(args, "pbp_root", None) or pbp_default),
        depth_root=Path(getattr(args, "depth_root", None) or depth_default),
        output_dir=Path(getattr(args, "output_dir", None) or out_default),
        years=parse_years(getattr(args, "years", None)),
        use_samples=bool(getattr(args, "use_samples", False)),
    )
    cfg.ensure()
    return cfg


def parse_years(s: Optional[str]) -> List[int]:
    if not s:
        return list(range(2020, 2026))
    if ":" in s:
        a, b = s.split(":", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in re.split(r"[ ,]+", s.strip()) if x]


# ----------------------------- Logging -----------------------------------

def setup_logging(out_dir: Path) -> None:
    log_path = out_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Logging to %s", log_path)


# ----------------------------- Utilities ---------------------------------

_TEMP_RE1 = re.compile(r"Temp:\s*(-?\d+)", re.I)
_TEMP_RE2 = re.compile(r"(-?\d+)\s*°?\s*F\b", re.I)
_WIND_RE1 = re.compile(r"Wind:\s*[A-Z]{0,3}\s*(\d+)\s*mph", re.I)
_WIND_RE2 = re.compile(r"(\d+)\s*mph", re.I)


def flexible_pick(cols: Iterable[str], targets: Iterable[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for t in targets:
        if t.lower() in lower:
            return lower[t.lower()]
    return None


def extract_temp(weather: object) -> float:
    if not isinstance(weather, str):
        return float("nan")
    m = _TEMP_RE1.search(weather) or _TEMP_RE2.search(weather)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r"\b(-?\d{1,3})\b", weather)
    if m:
        v = float(m.group(1))
        if -30 <= v <= 130:
            return v
    return float("nan")


def extract_wind_mph(weather: object) -> float:
    if not isinstance(weather, str):
        return float("nan")
    m = _WIND_RE1.search(weather) or _WIND_RE2.search(weather)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return float("nan")


def indoor_from_roof(val: object) -> int:
    s = str(val).strip().lower()
    # nflverse roof values often: 'outdoors', 'dome', 'outdoors (open)', 'closed'
    if s in {"outdoors", "outdoor"}:
        return 0
    return 1


def dist_band(d: int) -> str:
    if d <= 39:
        return "short"
    if d <= 49:
        return "mid"
    if d <= 59:
        return "long"
    return "xlong"


# ----------------------------- I/O Layer ---------------------------------

SAMPLE_PBP_CSV = Path("/mnt/data/play_by_play_2025.csv")  # uploaded sample
SAMPLE_DEPTH_CSV = Path("/mnt/data/depth_charts_2024.csv")  # uploaded sample


def load_pbp_year(cfg: Config, year: int) -> pd.DataFrame:
    """Load play-by-play for a given year. Prefer parquet; fallback to CSV sample when requested.
    Normalizes a minimal set of columns required by the pipeline.
    """
    logging.info("Loading PBP for %s", year)

    pq = cfg.pbp_root / f"play_by_play_{year}.parquet"
    csv = cfg.pbp_root / f"play_by_play_{year}.csv"

    df: Optional[pd.DataFrame] = None
    if pq.exists():
        df = pd.read_parquet(pq)
        logging.info("  loaded %s (%d rows)", pq, len(df))
    elif csv.exists():
        df = pd.read_csv(csv, low_memory=False)
        logging.info("  loaded %s (%d rows)", csv, len(df))
    elif cfg.use_samples and SAMPLE_PBP_CSV.exists() and year == 2025:
        df = pd.read_csv(SAMPLE_PBP_CSV, low_memory=False)
        logging.info("  loaded sample %s (%d rows)", SAMPLE_PBP_CSV, len(df))
    else:
        logging.warning("  MISSING pbp for %s (searched %s or %s)", year, pq, csv)
        return pd.DataFrame()

    df["season_year"] = year
    return df


def load_depth_year(cfg: Config, year: int) -> pd.DataFrame:
    """Load depth charts for a year (CSV). Only 2024 is guaranteed; 2025 may be missing."""
    logging.info("Loading depth charts for %s", year)
    fp = cfg.depth_root / f"depth_charts_{year}.csv"
    if fp.exists():
        df = pd.read_csv(fp, low_memory=False)
        logging.info("  loaded %s (%d rows)", fp, len(df))
        return df
    if cfg.use_samples and SAMPLE_DEPTH_CSV.exists() and year == 2024:
        df = pd.read_csv(SAMPLE_DEPTH_CSV, low_memory=False)
        logging.info("  loaded sample %s (%d rows)", SAMPLE_DEPTH_CSV, len(df))
        return df
    logging.warning("  MISSING depth charts for %s (looked in %s)", year, fp)
    return pd.DataFrame()


def load_pbp_many(cfg: Config) -> pd.DataFrame:
    frames = []
    for y in cfg.years:
        df = load_pbp_year(cfg, y)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No PBP data found for the requested years.")
    out = pd.concat(frames, ignore_index=True)
    logging.info("PBP combined: %d rows across %d season(s)", len(out), len(frames))
    return out


# -------------------------- Dataset Builders -----------------------------

def build_fg_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    play_type = flexible_pick(cols, {"play_type", "playtype", "play_type_nfl"})
    fg_result = flexible_pick(cols, {"field_goal_result", "fg_result", "fieldgoalresult"})
    kick_dist = flexible_pick(cols, {"kick_distance", "field_goal_distance", "kickdistance"})
    roof = flexible_pick(cols, {"roof"})
    weather = flexible_pick(cols, {"weather", "weather_detail", "weather_description"})
    fg_attempt = flexible_pick(cols, {"field_goal_attempt", "fg_attempt"})
    desc = flexible_pick(cols, {"desc", "description", "play_desc"})
    game_id = flexible_pick(cols, {"game_id", "gameid"})
    game_date = flexible_pick(cols, {"game_date", "gamedate", "game_day"})
    kicker_id = flexible_pick(cols, {"kicker_player_id", "kicker_id", "kickerid"})
    kicker_nm = flexible_pick(cols, {"kicker_player_name", "kicker", "kicker_name"})
    team = flexible_pick(cols, {"posteam", "pos_team", "offense_team"})

    # FG attempts mask
    if play_type is not None:
        fg_mask = df[play_type].astype(str).str.lower().eq("field_goal")
    elif fg_attempt is not None:
        fg_mask = pd.to_numeric(df[fg_attempt], errors="coerce").fillna(0).astype(int).eq(1)
    elif fg_result is not None:
        fg_mask = df[fg_result].notna()
    else:
        fg_mask = pd.Series(False, index=df.index)

    g = df.loc[fg_mask].copy()

    # Label: FG made
    if fg_result is not None:
        g["fg_made"] = g[fg_result].astype(str).str.lower().isin(["made", "good", "success"]).astype(int)
    elif desc is not None:
        g["fg_made"] = g[desc].astype(str).str.contains(r"\b(is )?good\b|\bmade\b", case=False, regex=True).astype(int)
    else:
        g["fg_made"] = np.nan

    # Distance
    if kick_dist is not None:
        g["distance"] = pd.to_numeric(g[kick_dist], errors="coerce")
    elif desc is not None:
        g["distance"] = pd.to_numeric(g[desc].str.extract(r"(\d+)\s*-\s*yard\s*field\s*goal", expand=False), errors="coerce")
    else:
        g["distance"] = np.nan

    # Env + Weather
    g["is_indoor"] = g[roof].apply(indoor_from_roof) if roof else 0
    if weather:
        g["temp_F"] = g[weather].apply(extract_temp)
        g["wind_mph"] = g[weather].apply(extract_wind_mph)
    else:
        g["temp_F"] = np.nan
        g["wind_mph"] = np.nan

    # IDs/dates
    g["game_id"] = g[game_id] if game_id else np.arange(len(g))
    g["kicker_id"] = g[kicker_id] if kicker_id else np.nan
    g["kicker_name"] = g[kicker_nm] if kicker_nm else g["kicker_id"]
    g["posteam"] = g[team] if team else np.nan

    if game_date:
        g["game_date"] = pd.to_datetime(df.loc[g.index, game_date], errors="coerce")
    else:
        g["game_date"] = pd.to_datetime(g.get("season", g["season_year"]).astype(str) + "-10-15", errors="coerce")

    # Clean
    g = g.dropna(subset=["distance"]).copy()
    g = g[g["fg_made"].isin([0, 1])]
    g["distance"] = g["distance"].astype(int)
    g = g[(g["distance"] >= 18) & (g["distance"] <= 68)]

    g["temp_F"] = pd.to_numeric(g["temp_F"], errors="coerce")
    g["wind_mph"] = pd.to_numeric(g["wind_mph"], errors="coerce")
    g["temp_F"] = g["temp_F"].fillna(g["temp_F"].median())
    g["wind_mph"] = g["wind_mph"].fillna(g["wind_mph"].median())

    g["kicker_id"] = g["kicker_id"].astype(str).fillna("unknown")

    # Final feature list (for Section 2 modeling):
    g["indoor_outdoor"] = np.where(g["is_indoor"] == 1, "indoor", "outdoor")
    return g.reset_index(drop=True)


def build_attempt_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    play_type = flexible_pick(cols, {"play_type", "playtype", "play_type_nfl"})
    down = flexible_pick(cols, {"down"})
    ytg = flexible_pick(cols, {"ydstogo", "yards_to_go", "yds_to_go"})
    yardline = flexible_pick(cols, {"yardline_100", "yardline"})
    roof = flexible_pick(cols, {"roof"})
    weather = flexible_pick(cols, {"weather", "weather_detail", "weather_description"})
    game_id = flexible_pick(cols, {"game_id", "gameid"})
    game_date = flexible_pick(cols, {"game_date", "gamedate", "game_day"})
    score_diff = flexible_pick(cols, {"score_differential", "score_diff"})
    team = flexible_pick(cols, {"posteam", "pos_team", "offense_team"})
    kicker_id = flexible_pick(cols, {"kicker_player_id", "kicker_id", "kickerid"})
    half_sec = flexible_pick(cols, {"half_seconds_remaining", "half_seconds"})

    if down is None or yardline is None or (play_type is None and score_diff is None):
        raise ValueError("Attempt frame requires down, yardline_100, and at least one of play_type or score_diff")

    att = df.copy()
    att = att[att[down] == 4].copy()

    att["distance"] = pd.to_numeric(att[yardline], errors="coerce") + 17
    att = att.dropna(subset=["distance"])
    att["distance"] = att["distance"].astype(int)
    att = att[(att["distance"] >= 18) & (att["distance"] <= 68)].copy()

    if play_type is not None:
        att["attempt_fg"] = att[play_type].astype(str).str.lower().eq("field_goal").astype(int)
    else:
        att["attempt_fg"] = 0

    att["is_indoor"] = att[roof].apply(indoor_from_roof) if roof else 0

    if weather:
        att["temp_F"] = att[weather].apply(extract_temp)
        att["wind_mph"] = att[weather].apply(extract_wind_mph)
    else:
        att["temp_F"], att["wind_mph"] = np.nan, np.nan
    att["temp_F"] = pd.to_numeric(att["temp_F"], errors="coerce").fillna(att["temp_F"].median())
    att["wind_mph"] = pd.to_numeric(att["wind_mph"], errors="coerce").fillna(att["wind_mph"].median())

    att["ydstogo"] = pd.to_numeric(att[ytg], errors="coerce").fillna(0) if ytg else 0
    att["score_diff"] = pd.to_numeric(att[score_diff], errors="coerce").fillna(0) if score_diff else 0
    att["score_diff"] = att["score_diff"].clip(-21, 21)

    if half_sec:
        att["half_sec"] = pd.to_numeric(att[half_sec], errors="coerce").fillna(900)
    else:
        att["half_sec"] = 900

    att["posteam"] = att[team].astype(str) if team else ""

    att["game_id"] = att[game_id] if game_id else np.arange(len(att))
    if game_date:
        att["game_date"] = pd.to_datetime(att[game_date], errors="coerce")
    else:
        year_col = flexible_pick(cols, {"season", "season_year"})
        if year_col:
            att["game_date"] = pd.to_datetime(att[year_col].astype(str) + "-10-15", errors="coerce")
        else:
            att["game_date"] = pd.NaT

    # (game, team) → mode kicker_id from rows where a kicker_id is present in original df
    if kicker_id:
        kick_rows = df.loc[df[kicker_id].notna(), [game_id or "game_id", team or "posteam", kicker_id]].copy()
        kick_rows.columns = ["game_id", "posteam", "kicker_id"]
        mode_map: Dict[Tuple[object, object], str] = (
            kick_rows.groupby(["game_id", "posteam"])['kicker_id']
            .agg(lambda s: s.astype(str).mode().iloc[0] if len(s) else np.nan)
            .to_dict()
        )
        att["kicker_id_game"] = [mode_map.get((g, t), np.nan) for g, t in zip(att["game_id"], att["posteam"])]
    else:
        att["kicker_id_game"] = np.nan

    att["indoor_outdoor"] = np.where(att["is_indoor"] == 1, "indoor", "outdoor")
    return att.reset_index(drop=True)


# ------------------------ Depth Chart → Roster ----------------------------

def _depth_to_roster_2024(depth_df: pd.DataFrame) -> pd.DataFrame:
    if depth_df.empty:
        return pd.DataFrame()
    cols = list(depth_df.columns)
    season = flexible_pick(cols, {"season"})
    week = flexible_pick(cols, {"week"})
    team = flexible_pick(cols, {"team", "team_abbr"})
    pos = flexible_pick(cols, {"position", "pos"})
    player = flexible_pick(cols, {"player", "player_name", "name"})
    player_id = flexible_pick(cols, {"gsis_id", "player_id", "nfl_id"})
    status = flexible_pick(cols, {"status", "depth_team_position", "depth_team"})

    df = depth_df.copy()
    if pos and team:
        df = df[df[pos].astype(str).str.upper().eq("K")]  # kickers only
    else:
        return pd.DataFrame()

    df["team_id"] = df[team].astype(str)
    df["kicker_id"] = df[player_id].astype(str) if player_id else df[player].astype(str)
    df["kicker_name"] = df[player].astype(str) if player else df["kicker_id"]
    df["status_raw"] = df[status].astype(str) if status else ""

    # Starter detection: prefer explicit starters; else depth order 1; else first
    starter_mask = df["status_raw"].str.contains("starter|first|1", case=False, regex=True)
    pref = df[starter_mask].copy()
    if pref.empty:
        pref = df.copy()

    # pick most recent week per team as canonical starter
    if week:
        pref = pref.sort_values(by=[team, week])
        starter_map = pref.groupby("team_id")["kicker_id"].last().to_dict()
    else:
        starter_map = pref.groupby("team_id")["kicker_id"].first().to_dict()

    # status flags (very naive harmonization)
    def _status_flag(s: str) -> str:
        s = s.lower()
        if any(k in s for k in ["inactive", "ir", "practice", "ps"]):
            return "inactive"
        return "active"

    df["status"] = df["status_raw"].map(_status_flag)

    # emit roster rows per team
    roster_rows = []
    for t, g in df.groupby("team_id"):
        ks = (
            g.sort_values(by=[week] if week else [])
            .drop_duplicates(subset=["kicker_id"])  # one row per kicker
        )
        kickers = [
            {"kicker_id": rid, "name": nm, "status": st}
            for rid, nm, st in zip(ks["kicker_id"], ks["kicker_name"], ks["status"])  # type: ignore
        ]
        roster_rows.append({
            "team_id": t,
            "team_name": t,
            "starter_kicker_id": starter_map.get(t),
            "kickers": kickers,
        })
    return pd.json_normalize(roster_rows)


def _pbp_fallback_roster(df_pbp: pd.DataFrame, year: int) -> pd.DataFrame:
    """When depth chart is missing (e.g., 2025), infer team starter as **mode kicker** per team
    across the season (using FG plays)."""
    if df_pbp.empty:
        return pd.DataFrame()

    cols = list(df_pbp.columns)
    team = flexible_pick(cols, {"posteam", "pos_team", "offense_team"})
    kicker_id = flexible_pick(cols, {"kicker_player_id", "kicker_id", "kickerid"})
    kicker_nm = flexible_pick(cols, {"kicker_player_name", "kicker", "kicker_name"})
    play_type = flexible_pick(cols, {"play_type", "playtype", "play_type_nfl"})

    df = df_pbp.copy()
    if play_type:
        df = df[df[play_type].astype(str).str.lower().eq("field_goal")]
    df = df[[team or "posteam", kicker_id or "kicker_id", kicker_nm or "kicker_name"]].dropna()
    df.columns = ["team_id", "kicker_id", "kicker_name"]

    mode_kicker = df.groupby("team_id")["kicker_id"].agg(lambda s: s.astype(str).mode().iloc[0]).to_dict()

    # assemble roster list with all observed kickers marked active
    roster_rows = []
    for t, g in df.groupby("team_id"):
        ks = g.drop_duplicates(subset=["kicker_id"])  # unique kickers observed
        kickers = [
            {"kicker_id": rid, "name": nm, "status": "active"}
            for rid, nm in zip(ks["kicker_id"], ks["kicker_name"])  # type: ignore
        ]
        roster_rows.append({
            "team_id": t,
            "team_name": t,
            "starter_kicker_id": mode_kicker.get(t),
            "kickers": kickers,
        })
    return pd.json_normalize(roster_rows)


def build_roster(cfg: Config, pbp_df: pd.DataFrame) -> Dict:
    depth24 = load_depth_year(cfg, 2024)
    roster24 = _depth_to_roster_2024(depth24)

    roster25 = _pbp_fallback_roster(pbp_df[pbp_df["season_year"] == 2025], 2025)

    roster_all = []
    if not roster24.empty:
        roster_all.extend(roster24.to_dict(orient="records"))
    if not roster25.empty:
        # merge 2025 fallback into 2024 where missing
        have = {r["team_id"] for r in roster_all}
        for r in roster25.to_dict(orient="records"):
            if r["team_id"] not in have:
                roster_all.append(r)
    return {"teams": roster_all}


# ---------------------------- Validation ---------------------------------

def validate_fg(g: pd.DataFrame) -> List[str]:
    msgs: List[str] = []
    if g["distance"].lt(18).any() or g["distance"].gt(68).any():
        msgs.append("FG distance out of bounds present")
    if g["fg_made"].isna().any():
        msgs.append("Missing fg_made labels found")
    for c in ["temp_F", "wind_mph"]:
        if g[c].isna().any():
            msgs.append(f"Missing {c} after fillna")
    return msgs


def validate_attempt(att: pd.DataFrame) -> List[str]:
    msgs: List[str] = []
    if att["distance"].lt(18).any() or att["distance"].gt(68).any():
        msgs.append("Attempt distance out of bounds present")
    if att["attempt_fg"].isna().any():
        msgs.append("Missing attempt label")
    return msgs


# ------------------------------ Main -------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="FG Pipeline — Section 1/2 (data + roster)")
    parser.add_argument("--pbp-root", type=str, default=None)
    parser.add_argument("--depth-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--years", type=str, default=None, help="e.g., '2020:2025' or '2022 2023 2024'")
    parser.add_argument("--use-samples", action="store_true", help="Use uploaded sample CSVs for missing files")
    args = parser.parse_args(argv)

    cfg = default_config(args)
    setup_logging(cfg.output_dir)
    logging.info("Config: %s", cfg)

    # Load
    pbp_df = load_pbp_many(cfg)

    # Build curated datasets
    fg_df = build_fg_frame(pbp_df)
    att_df = build_attempt_frame(pbp_df)

    # Build roster (depth charts 2024; pbp fallback 2025)
    roster = build_roster(cfg, pbp_df)

    # Validate & persist
    msgs_fg = validate_fg(fg_df)
    msgs_att = validate_attempt(att_df)

    fg_path = cfg.output_dir / "curated_fg.parquet"
    att_path = cfg.output_dir / "curated_attempt.parquet"
    roster_path = cfg.output_dir / "nfl_kicker_roster.json"
    report_path = cfg.output_dir / "data_report.txt"

    fg_df.to_parquet(fg_path, index=False)
    att_df.to_parquet(att_path, index=False)
    with open(roster_path, "w", encoding="utf-8") as f:
        json.dump(roster, f)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("== Data Report (Section 1) ==\n")
        f.write(f"FG rows: {len(fg_df):,}\n")
        f.write(f"Attempt rows: {len(att_df):,}\n")
        if msgs_fg:
            f.write("\nFG issues:\n  - " + "\n  - ".join(msgs_fg) + "\n")
        if msgs_att:
            f.write("\nAttempt issues:\n  - " + "\n  - ".join(msgs_att) + "\n")
        f.write("\nColumns (FG):\n  - " + ", ".join(fg_df.columns) + "\n")
        f.write("\nColumns (ATT):\n  - " + ", ".join(att_df.columns) + "\n")

    logging.info("Saved: %s", fg_path)
    logging.info("Saved: %s", att_path)
    logging.info("Saved: %s", roster_path)
    logging.info("Saved: %s", report_path)


if __name__ == "__main__":
    main()
