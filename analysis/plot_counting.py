#!/usr/bin/env python3
"""Card-counting charts from an experiment arm's count_profile.csv.

Usage:
    python analysis/plot_counting.py experiments/002-card-counting/confirm/learned-counter-50m \\
        --output docs/images/counting_edge.png

Two panels sharing the true-count axis (two measures, so two panels, never a
second y-axis):
  1. Edge by true count: per-unit result of flat basic strategy and of the
     learned agent's play, pooled over seeds, with 95% intervals.
  2. Bet by true count: textbook Hi-Lo's ramp against the bets the agent chose
     from its own measured edge.
Textbook is blue and the learned agent orange in both panels.
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Reference palette, light surface: categorical slots 1 and 2 (validated as a
# pair: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface).
TEXTBOOK = "#2a78d6"
LEARNED = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

PER_ROUND_SD = 1.15  # per-round reward SD in blackjack, for the interval
HILO_RAMP = {-3: 1, -2: 1, -1: 1, 0: 1, 1: 1, 2: 2, 3: 4, 4: 6, 5: 8, 6: 8}


def tc_label(tc: int) -> str:
    if tc <= -3:
        return "≤−3"
    if tc >= 6:
        return "≥+6"
    return f"{tc:+d}".replace("-", "−") if tc else "0"


def pooled(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    p = df[df["policy"] == policy].copy()
    p["won"] = p["mean_reward"] * p["rounds"]
    g = p.groupby("true_count").agg(rounds=("rounds", "sum"), won=("won", "sum")).reset_index()
    g["edge"] = g["won"] / g["rounds"]
    g["ci"] = 1.96 * PER_ROUND_SD / g["rounds"].map(math.sqrt)
    return g.sort_values("true_count")


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm", type=Path, help="folder holding count_profile.csv")
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    csv = args.arm / "count_profile.csv"
    if not csv.exists():
        print(f"No count_profile.csv in {args.arm}", file=sys.stderr)
        return 1
    df = pd.read_csv(csv)
    book = pooled(df, "basic_strategy")
    agent = pooled(df, "agent")
    tcs = list(book["true_count"])
    labels = [tc_label(t) for t in tcs]

    # The bet each seed's agent chose at each count; show the most common one.
    bets = (df[df["policy"] == "agent"].groupby("true_count")["learned_bet"]
            .agg(lambda s: s.mode().iloc[0]).reindex(tcs))

    fig, (edge_ax, bet_ax) = plt.subplots(
        1, 2, figsize=(12, 4.6), dpi=150, facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- Edge by true count ----
    style(edge_ax)
    edge_ax.axhline(0, color=INK_2, linewidth=1)
    series = ((book, TEXTBOOK, "Basic strategy", -0.08),
              (agent, LEARNED, "Learned agent", 0.08))
    # End-of-line labels, nudged apart so they never sit on each other.
    ends = sorted(((g["edge"].iloc[-1], name) for g, _, name, _ in series), reverse=True)
    nudge = {name: (9 if rank == 0 else -9) for rank, (_, name) in enumerate(ends)}
    for g, color, name, dx in series:
        x = [i + dx for i in range(len(g))]
        y = (g["edge"] * 100).tolist()
        err = (g["ci"] * 100).tolist()
        edge_ax.errorbar(x, y, yerr=err, color=color, linewidth=2, marker="o",
                         markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                         elinewidth=1.2, capsize=0, label=name)
        edge_ax.annotate(name, (x[-1], y[-1]), xytext=(10, nudge[name]),
                         textcoords="offset points", va="center", fontsize=10, color=INK)
    edge_ax.set_xticks(range(len(labels)), labels)
    edge_ax.set_xlim(-0.5, len(labels) + 1.3)
    edge_ax.set_xlabel("Hi-Lo true count before the deal", color=INK_2)
    edge_ax.set_ylabel("Result per unit bet (%)", color=INK_2)
    edge_ax.set_title("The player's edge rises with the count", loc="left",
                      color=INK, fontsize=12, fontweight="bold")
    edge_ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_2)

    # ---- Bet by true count ----
    style(bet_ax)
    x = list(range(len(tcs)))
    ramp = [HILO_RAMP[t] for t in tcs]
    bet_ax.step(x, ramp, where="mid", color=TEXTBOOK, linewidth=2, label="Hi-Lo ramp")
    bet_ax.step(x, bets.tolist(), where="mid", color=LEARNED, linewidth=2,
                linestyle=(0, (4, 2)), label="Learned bets")
    bet_ax.plot(x, ramp, "o", color=TEXTBOOK, markersize=6, markeredgecolor=SURFACE)
    bet_ax.plot(x, bets.tolist(), "o", color=LEARNED, markersize=6, markeredgecolor=SURFACE)
    bet_ax.set_xticks(x, labels)
    bet_ax.set_yticks([1, 2, 4, 6, 8])
    bet_ax.set_ylim(0, 9)
    bet_ax.set_xlabel("Hi-Lo true count before the deal", color=INK_2)
    bet_ax.set_ylabel("Units bet", color=INK_2)
    bet_ax.set_title("How much each one bets", loc="left", color=INK,
                     fontsize=12, fontweight="bold")
    bet_ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_2)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
