#!/usr/bin/env python3
import json, pathlib, re, sys
import pandas as pd

DEPTH_DIR = pathlib.Path("depth_charts/kickers")
OUT_JSON  = pathlib.Path("nfl_kicker_roster.json")

NFL = {
  "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC",
  "LAC","LAR","MIA","MIN","NE","NO","NYG","NYJ","LV","PHI","PIT","SEA","SF","TB","TEN","WAS"
}

def latest_csv():
  if not DEPTH_DIR.exists():
    return None
  files = sorted(DEPTH_DIR.glob("kickers_depth_charts_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
  return files[0] if files else None

def normalize_team(x): return (str(x) or "").strip().upper()

def is_active(status: str) -> bool:
  s = (status or "").lower()
  # treat doubtful/ques as active; exclude clear inactive signals
  return not any(tag in s for tag in ["inactive", "out", "ir", "retired"])

def row_id(row):
  for c in ("gsis_id","player_id","pfr_id"):
    v = (str(row.get(c) or "").strip())
    if v: return v
  # fallback deterministic id
  nm = re.sub(r"[^A-Za-z0-9]+","_", str(row.get("full_name","")).strip())
  return f"{row.get('club_code','UNK')}_{nm}" or nm or "unknown"

def main():
  src = latest_csv()
  if not src or not src.exists():
    print("ERROR: No kicker depth chart CSV found in depth_charts/kickers/", file=sys.stderr)
    sys.exit(2)

  df = pd.read_csv(src)
  # Guard: only valid NFL teams
  df["club_code"] = df["club_code"].map(normalize_team)
  df = df[df["club_code"].isin(NFL)].copy()

  # Starter = depth_chart_order == 1
  df["__starter"] = (df["depth_chart_order"].fillna(99).astype(int) == 1)
  df["__active"]  = df.get("status", "").apply(is_active)

  records = []
  for team, g in df.groupby("club_code"):
    kickers = []
    g = g.sort_values(["__starter","full_name"], ascending=[False, True])
    for _, r in g.iterrows():
      kid = row_id(r)
      kickers.append({
        "id": kid,
        "name": str(r.get("full_name","")).strip() or "Kicker",
        "starter": bool(r["__starter"]),
        "active":  bool(r["__active"])
      })
    if kickers:
      records.append({"team": team, "kickers": kickers})

  OUT_JSON.write_text(json.dumps(records, indent=2))
  print(f"Wrote {OUT_JSON.resolve()} for {len(records)} teams, {sum(len(r['kickers']) for r in records)} kickers.")

if __name__ == "__main__":
  main()
