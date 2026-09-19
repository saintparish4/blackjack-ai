#!/usr/bin/env python3
"""Rank experiment arms by EV from their summary.csv files.

Usage:
    python analysis/leaderboard.py experiments/001-double-down/explore
    python analysis/leaderboard.py experiments/001-double-down/confirm --markdown

Each subfolder is one arm written by ./build/experiment (summary.csv holds the
across-seed mean, sd, t-based 95% CI, min and max of every metric).
"""

import argparse
import csv
import sys
from pathlib import Path


def load(arm_dir: Path) -> dict:
    with open(arm_dir / "summary.csv", newline="") as f:
        rows = {r["metric"]: r for r in csv.DictReader(f)}
    return {"arm": arm_dir.name, "seeds": int(rows["agent_ev"]["seeds"]), **rows}


def fmt(row: dict, metric: str, scale: float = 1.0, digits: int = 4) -> str:
    if metric not in row:
        return "n/a"
    m = float(row[metric]["mean"]) * scale
    ci = float(row[metric]["ci95"]) * scale
    return f"{m:+.{digits}f} ± {ci:.{digits}f}" if scale == 1.0 else f"{m:.{digits}f} ± {ci:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--markdown", action="store_true", help="Markdown table")
    args = parser.parse_args()

    arms = [load(d) for d in sorted(args.folder.iterdir()) if (d / "summary.csv").exists()]
    if not arms:
        print(f"No summary.csv under {args.folder}", file=sys.stderr)
        return 1
    arms.sort(key=lambda r: float(r["agent_ev"]["mean"]), reverse=True)

    header = ["Arm", "Seeds", "Agent EV / round", "Book EV / round",
              "Gap recovered %", "Two-card %", "Pairs %", "Overall %"]
    rows = [[a["arm"], str(a["seeds"]), fmt(a, "agent_ev"), fmt(a, "book_ev"),
             fmt(a, "gap_recovered", 100, 1), fmt(a, "two_card_accuracy", 100, 1),
             fmt(a, "pair_accuracy", 100, 1), fmt(a, "accuracy", 100, 1)]
            for a in arms]

    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        for r in rows:
            print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
