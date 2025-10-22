#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_config.py — Central configuration, paths, logging, helpers.

- Parse year ranges like "2020:2025" or "2020,2022,2024".
- Provide a Config dataclass with sensible defaults for your machine.
- Lightweight utilities reused by downstream steps (logging, weather parsing,
  stadiums loader, air density, wind-direction parsing).

This file has ZERO side effects unless executed as __main__ (then it prints the resolved config).
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ------------------------ Defaults (tuned to your setup) ------------------------

# NOTE: These defaults are only used if corresponding CLI flags are omitted.
DEFAULT_PBP_ROOT   = Path(r"C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp")
DEFAULT_DEPTH_ROOT = Path(r"C:\Users\awolk\Documents\NFELO\Other Data\nflverse\depth_charts")
DEFAULT_PROJECT    = Path(r"C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp\fg_project")

DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT / "artifacts"   # curated + features (not committed)
DEFAULT_UI_DIR     = DEFAULT_PROJECT / "ui"          # index.html + JSONs (committed)
DEFAULT_ASSETS_DIR = DEFAULT_PROJECT / "assets"      # e.g., stadiums.csv (optional)

DEFAULT_YEARS_STR  = "2020:2025"
DEFAULT_LOGLEVEL   = "INFO"


# ------------------------ Small utility helpers ------------------------

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def parse_years(spec: Optional[str]) -> List[int]:
    """
    Parse '2020:2025' ⇒ [2020,2021,2022,2023,2024,2025]
          '2020,2022,2024' ⇒ [2020,2022,2024]
    """
    if not spec:
        spec = DEFAULT_YEARS_STR
    spec = str(spec).strip()
    if ":" in spec:
        a, b = spec.split(":")
        a, b = int(a), int(b)
        if b < a:
            raise ValueError(f"years range invalid: {spec}")
        return list(range(a, b + 1))
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("no years parsed from spec")
    return sorted(set(out))

def setup_logging(level: str = DEFAULT_LOGLEVEL) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    # Keep logs clean for repeated runs
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# ------------------------ Weather parsing ------------------------

_TEMP_RE1 = re.compile(r"Temp:\s*(-?\d+)", re.I)
_TEMP_RE2 = re.compile(r"(-?\d+)\s*°?\s*F\b", re.I)
_WIND_RE_MPH = re.compile(r"(\d+)\s*mph", re.I)
# e.g., "Wind: NE 12 mph", "Wind: W 8 mph gusts 15"
_WIND_DIR_CARD = re.compile(r"Wind:\s*([NSEW]{1,3})", re.I)

_CARD_TO_DEG = {
    # Meteorological convention: direction wind is FROM (0=North, 90=East)
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

def extract_temp_f(weather: object) -> float:
    if not isinstance(weather, str):
        return float("nan")
    m = _TEMP_RE1.search(weather) or _TEMP_RE2.search(weather)
    if m:
        try: return float(m.group(1))
        except Exception: pass
    # Fallback: any plausible integer
    m = re.search(r"\b(-?\d{1,3})\b", weather)
    if m:
        v = float(m.group(1))
        if -30 <= v <= 130:
            return v
    return float("nan")

def extract_wind_mph(weather: object) -> float:
    if not isinstance(weather, str):
        return float("nan")
    m = _WIND_RE_MPH.search(weather)
    if m:
        try: return float(m.group(1))
        except Exception: pass
    return float("nan")

def extract_wind_dir_deg(weather: object) -> float:
    """Return degrees (0..360) meteorological FROM-direction; NaN if unknown."""
    if not isinstance(weather, str):
        return float("nan")
    m = _WIND_DIR_CARD.search(weather)
    if not m:
        return float("nan")
    card = m.group(1).upper()
    return float(_CARD_TO_DEG.get(card, float("nan")))

# ------------------------ Physics helpers ------------------------

def f_to_kelvin(F: float) -> float:
    return (F - 32.0) * 5.0 / 9.0 + 273.15

def std_air_density_kg_m3(alt_m: float, temp_K: float) -> float:
    """
    Approximate air density using International Standard Atmosphere (ISA):
    - Troposphere up to ~11km.
    Inputs:
      alt_m: altitude above sea level (meters)
      temp_K: ambient temperature (Kelvin)
    Returns:
      rho [kg/m^3]
    """
    # Constants
    p0 = 101325.0       # sea-level standard atmospheric pressure (Pa)
    T0 = 288.15         # sea-level standard temperature (K)
    g  = 9.80665        # m/s^2
    L  = 0.0065         # temperature lapse rate (K/m)
    R  = 8.3144598      # universal gas constant (J/(mol·K))
    M  = 0.0289644      # molar mass of dry air (kg/mol)
    # Pressure at altitude
    if alt_m < 0: alt_m = 0.0
    p = p0 * (1.0 - (L * alt_m) / T0) ** (g * M / (R * L))
    # Density via ideal gas law  ρ = p / (R_specific * T)  with R_specific = R/M
    rho = p * M / (R * max(temp_K, 1.0))
    return float(rho)

def wind_components_mps(wind_mph: float, from_dir_deg: float, axis_deg: float) -> Tuple[float, float]:
    """
    Resolve wind into headwind and crosswind components relative to an axis (uprights normal).

    Args:
      wind_mph: scalar speed (mph)
      from_dir_deg: meteorological FROM direction (0=N, 90=E). NaN => components NaN.
      axis_deg: axis the ball travels toward (degrees clockwise from North).
                Example: 'uprights azimuth' (direction TO uprights).

    Returns:
      (head_mps, cross_mps), signed.
      head_mps > 0 means wind opposes ball (true headwind).
      cross_mps > 0 means wind from right-to-left across the axis.
    """
    if not (np.isfinite(wind_mph) and np.isfinite(from_dir_deg) and np.isfinite(axis_deg)):
        return (float("nan"), float("nan"))
    # Convert mph -> m/s
    v = wind_mph * 0.44704
    # Convert FROM to TO direction
    to_dir = (from_dir_deg + 180.0) % 360.0
    # Angle difference between wind TO direction and travel axis
    delta = math.radians((to_dir - axis_deg + 540.0) % 360.0 - 180.0)
    head = v * math.cos(delta) * (-1.0)  # positive => headwind (opposes travel)
    cross = v * math.sin(delta)          # positive => R->L
    return (float(head), float(cross))

# ------------------------ Stadiums ------------------------

def load_stadiums(assets_dir: Path) -> pd.DataFrame:
    """
    Try to load stadiums.csv from assets_dir. Expected columns (case-insensitive):
      team, stadium, city, altitude_m, uprights_azimuth_deg, roof, surface
    Returns empty DataFrame if missing.
    """
    assets_dir = Path(assets_dir)
    csv = assets_dir / "stadiums.csv"
    if not csv.exists():
        logging.getLogger("config").warning(
            "assets/stadiums.csv not found at %s — altitude/azimuth features will be NaN/assumed.",
            csv
        )
        return pd.DataFrame()
    df = pd.read_csv(csv)
    # Normalize columns
    lower = {c.lower(): c for c in df.columns}
    def col(name_set: Iterable[str]) -> Optional[str]:
        for n in name_set:
            if n in lower: return lower[n]
        return None

    # Basic renames if needed
    rename_map = {}
    if col({"team"}) and col({"team"}) != "team": rename_map[col({"team"})] = "team"
    if col({"stadium"}) and col({"stadium"}) != "stadium": rename_map[col({"stadium"})] = "stadium"
    if col({"altitude_m", "alt_m"}) and col({"altitude_m", "alt_m"}) != "altitude_m":
        rename_map[col({"altitude_m","alt_m"})] = "altitude_m"
    if col({"uprights_azimuth_deg", "upright_azimuth_deg", "axis_deg"}) and \
       col({"uprights_azimuth_deg", "upright_azimuth_deg", "axis_deg"}) != "uprights_azimuth_deg":
        rename_map[col({"uprights_azimuth_deg", "upright_azimuth_deg", "axis_deg"})] = "uprights_azimuth_deg"
    if col({"roof"}) and col({"roof"}) != "roof": rename_map[col({"roof"})] = "roof"
    if col({"surface"}) and col({"surface"}) != "surface": rename_map[col({"surface"})] = "surface"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

# ------------------------ Config dataclass ------------------------

@dataclass
class Config:
    pbp_root: Path
    depth_root: Path
    output_dir: Path
    ui_dir: Path
    assets_dir: Path
    years: List[int]
    log_level: str = DEFAULT_LOGLEVEL

    @classmethod
    def from_args(cls,
                  pbp_root: Optional[str] = None,
                  depth_root: Optional[str] = None,
                  output_dir: Optional[str] = None,
                  ui_dir: Optional[str] = None,
                  assets_dir: Optional[str] = None,
                  years: Optional[str] = None,
                  log_level: Optional[str] = None) -> "Config":
        cfg = cls(
            pbp_root   = Path(pbp_root or DEFAULT_PBP_ROOT),
            depth_root = Path(depth_root or DEFAULT_DEPTH_ROOT),
            output_dir = ensure_dir(Path(output_dir or DEFAULT_OUTPUT_DIR)),
            ui_dir     = ensure_dir(Path(ui_dir or DEFAULT_UI_DIR)),
            assets_dir = ensure_dir(Path(assets_dir or DEFAULT_ASSETS_DIR)),
            years      = parse_years(years or DEFAULT_YEARS_STR),
            log_level  = (log_level or DEFAULT_LOGLEVEL).upper(),
        )
        return cfg

    def to_json(self) -> str:
        return json.dumps({
            "pbp_root":   str(self.pbp_root),
            "depth_root": str(self.depth_root),
            "output_dir": str(self.output_dir),
            "ui_dir":     str(self.ui_dir),
            "assets_dir": str(self.assets_dir),
            "years":      self.years,
            "log_level":  self.log_level,
        }, indent=2)


# ------------------------ CLI (show resolved config) ------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Show resolved FG project config.")
    ap.add_argument("--pbp-root")
    ap.add_argument("--depth-root")
    ap.add_argument("--output-dir")
    ap.add_argument("--ui-dir")
    ap.add_argument("--assets-dir")
    ap.add_argument("--years", help='e.g. "2020:2025" or "2020,2022,2024"')
    ap.add_argument("--log-level", default=DEFAULT_LOGLEVEL)
    args = ap.parse_args()

    cfg = Config.from_args(
        pbp_root=args.pbp_root,
        depth_root=args.depth_root,
        output_dir=args.output_dir,
        ui_dir=args.ui_dir,
        assets_dir=args.assets_dir,
        years=args.years,
        log_level=args.log_level,
    )
    setup_logging(cfg.log_level)
    logging.getLogger("config").info("Resolved config:\n%s", cfg.to_json())
    print(cfg.to_json())
