#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K_cli.py — One CLI to run the whole pipeline or individual steps.

Usage examples:
  py -3 pipeline\K_cli.py run-all --pbp-root "..." --depth-root "..." --years 2020:2025
  py -3 pipeline\K_cli.py run-core --pbp-root "..." --depth-root "..." --years 2020:2025
  py -3 pipeline\K_cli.py step G --pbp-root "..." --depth-root "..." --years 2020:2025

Steps (letters):
  A: config (module only, not run)
  B: ingest
  C: features
  D: make (GAM)
  E: make (GBM residual)
  F: kicker deltas
  G: attempt model
  H: coupling & sensitivity
  I: calibration eval
  J: export grids & manifest
"""

from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

DEFAULT_OUTPUT = ".\\artifacts"
DEFAULT_UI     = ".\\ui"
DEFAULT_STAD   = r"C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp\team_stadiums.csv"

def run(cmd: list[str]) -> int:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, shell=False)

def years_arg(y: str) -> str:
    # support "2020:2025" or comma lists
    return y

def main():
    ap = argparse.ArgumentParser(description="Pipeline Orchestrator")
    ap.add_argument("--pbp-root", required=True)
    ap.add_argument("--depth-root", required=True)
    ap.add_argument("--years", default="2020:2025")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--ui-dir", default=DEFAULT_UI)
    ap.add_argument("--stadiums-csv", default=DEFAULT_STAD)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bags-make", type=int, default=24)
    ap.add_argument("--bags-attempt", type=int, default=14)
    ap.add_argument("--continue-on-error", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run-all", help="Run everything B→J")
    sub.add_parser("run-core", help="Run B→H then J (skip I)")
    sp = sub.add_parser("step", help="Run a single step by letter (B..J)")
    sp.add_argument("letter", choices=list("BCDEFGHIJ"))

    args = ap.parse_args()

    py = sys.executable
    P = Path("pipeline")

    def step_B():
        return run([py, str(P/"B_ingest.py"),
                    "--pbp-root", args.pbp_root,
                    "--depth-root", args.depth_root,
                    "--output-dir", args.output_dir,
                    "--ui-dir", args.ui_dir,
                    "--years", years_arg(args.years)])

    def step_C():
        return run([py, str(P/"C_features.py"),
                    "--output-dir", args.output_dir,
                    "--assets-dir", ".\\assets",
                    "--stadiums-csv", args.stadiums_csv])

    def step_D():
        return run([py, str(P/"D_make_physics_gam.py"),
                    "--output-dir", args.output_dir, "--log-level", "INFO"])

    def step_E():
        return run([py, str(P/"E_make_gbm_residual.py"),
                    "--output-dir", args.output_dir, "--ui-dir", args.ui_dir,
                    "--bags", str(args.bags_make), "--seed", str(args.seed), "--log-level", "INFO"])

    def step_F():
        return run([py, str(P/"F_kicker_hier_bayes.py"),
                    "--output-dir", args.output_dir, "--ui-dir", args.ui_dir])

    def step_G():
        return run([py, str(P/"G_attempt_model.py"),
                    "--output-dir", args.output_dir, "--ui-dir", args.ui_dir,
                    "--bags", str(args.bags_attempt), "--seed", str(args.seed)])

    def step_H():
        return run([py, str(P/"H_coupling_and_wp.py"),
                    "--output-dir", args.output_dir, "--ui-dir", args.ui_dir])

    def step_I():
        return run([py, str(P/"I_calibration_eval.py"),
                    "--output-dir", args.output_dir])

    def step_J():
        return run([py, str(P/"J_export_grids.py"),
                    "--output-dir", args.output_dir, "--ui-dir", args.ui_dir, "--rebuild", "all"])

    steps = {
        "B": step_B, "C": step_C, "D": step_D, "E": step_E,
        "F": step_F, "G": step_G, "H": step_H, "I": step_I, "J": step_J
    }

    def run_seq(seq: list[str]) -> int:
        for s in seq:
            rc = steps[s]()
            if rc != 0 and not args.continue_on_error:
                print(f"!! Step {s} failed with code {rc}. Stopping.", flush=True)
                return rc
        return 0

    if args.cmd == "step":
        sys.exit(run_seq([args.letter]))
    elif args.cmd == "run-all":
        sys.exit(run_seq(list("BCDEFGHIJ")))
    elif args.cmd == "run-core":
        sys.exit(run_seq(list("BCDEFGHJ")))

if __name__ == "__main__":
    main()
