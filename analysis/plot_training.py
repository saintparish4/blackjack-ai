#!/usr/bin/env python3
"""Blackjack AI training visualization.

Reads CSV training logs and Q-table export, generates 5 publication-quality plots.

Usage:
    python plot_training.py logs/training_*.csv [--output plots/]
    python plot_training.py logs/training_20240101_120000.csv --output results/
    python plot_training.py logs/*.csv --qtable analysis/q_table.csv --output plots/

CSV columns (training log):
    episode, elapsed_sec, win_rate, loss_rate, push_rate,
    avg_reward, bust_rate, epsilon, states_learned

CSV columns (Q-table):
    player_total, dealer_card, usable_ace,
    Q_HIT, Q_STAND, Q_DOUBLE, Q_SPLIT, Q_SURRENDER
"""

import argparse
import glob as glob_mod
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---- constants ----------------------------------------------------------------

# Maps Q-table column suffix to single-char abbreviation shown in cells
ACTION_CHARS: dict[str, str] = {
    "HIT": "H",
    "STAND": "S",
    "DOUBLE": "D",
    "SPLIT": "P",
    "SURRENDER": "R",
}

BASIC_STRATEGY_WIN_RATE = 43.0  # approximate target reference line (%)

FIGURE_DPI = 150
FIGURE_SIZE_WIDE = (11, 5)
FIGURE_SIZE_HEATMAP = (12, 9)

# ---- data loading -------------------------------------------------------------


def load_logs(csv_files: list[str]) -> pd.DataFrame:
    """Load and concatenate training log CSVs.

    Multiple runs are merged and sorted by episode number, which lets you
    visualise a resumed training session as a single continuous curve.
    """
    frames: list[pd.DataFrame] = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as exc:
            print(f"  Warning: could not read '{path}': {exc}", file=sys.stderr)

    if not frames:
        print("Error: no valid log files loaded.", file=sys.stderr)
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True).sort_values("episode").reset_index(drop=True)
    return data


# ---- axis helpers -------------------------------------------------------------


def _episode_formatter(x: float, _pos: int) -> str:
    """Format episode counts as 500K or 1.2M for readability."""
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return str(int(x))


def _apply_episode_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_episode_formatter))


def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)
    _apply_episode_axis(ax)


def _save(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---- individual plots ---------------------------------------------------------


def plot_learning_curve(data: pd.DataFrame, output_dir: str) -> None:
    """Win / loss / push rates over episodes with a basic-strategy reference line."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    ax.plot(data["episode"], data["win_rate"] * 100,
            color="#2ecc71", linewidth=1.8, label="Win rate")
    ax.plot(data["episode"], data["loss_rate"] * 100,
            color="#e74c3c", linewidth=1.4, alpha=0.8, label="Loss rate")
    ax.plot(data["episode"], data["push_rate"] * 100,
            color="#3498db", linewidth=1.2, alpha=0.6, label="Push rate")
    ax.axhline(BASIC_STRATEGY_WIN_RATE, color="#7f8c8d", linestyle="--",
               linewidth=1.2, label=f"Basic strategy (~{BASIC_STRATEGY_WIN_RATE}%)")

    ax.set_ylim(0, 65)
    _style_axes(ax, "Episode", "Rate (%)", "Learning Curve — Win / Loss / Push Rates")

    _save(fig, os.path.join(output_dir, "learning_curve.png"))


def plot_reward_curve(data: pd.DataFrame, output_dir: str) -> None:
    """Average reward per episode with a zero-reward reference line."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    ax.plot(data["episode"], data["avg_reward"],
            color="#9b59b6", linewidth=1.8, label="Avg reward")
    ax.axhline(0, color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.6)

    _style_axes(ax, "Episode", "Average Reward", "Reward Curve — Average Reward per Evaluation")

    _save(fig, os.path.join(output_dir, "reward_curve.png"))


def plot_epsilon_decay(data: pd.DataFrame, output_dir: str) -> None:
    """Epsilon (exploration rate) decay over the training run."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    ax.plot(data["episode"], data["epsilon"],
            color="#f39c12", linewidth=1.8, label="Epsilon (exploration rate)")
    ax.set_ylim(-0.02, 1.05)

    _style_axes(ax, "Episode", "Epsilon", "Epsilon Decay — Exploration vs Exploitation")

    _save(fig, os.path.join(output_dir, "epsilon_decay.png"))


def plot_state_coverage(data: pd.DataFrame, output_dir: str) -> None:
    """Number of Q-table states visited over the training run."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    ax.plot(data["episode"], data["states_learned"],
            color="#1abc9c", linewidth=1.8, label="States learned")

    _style_axes(ax, "Episode", "States Learned", "State Coverage — Q-Table Exploration Progress")

    _save(fig, os.path.join(output_dir, "state_coverage.png"))


def plot_q_heatmap(qtable_path: str, output_dir: str) -> None:
    """Best-Q-value heatmap for hard totals with action labels in each cell."""
    if not os.path.exists(qtable_path):
        print(f"  Skipping Q-heatmap: '{qtable_path}' not found.")
        return

    try:
        df = pd.read_csv(qtable_path)
    except Exception as exc:
        print(f"  Skipping Q-heatmap: could not read '{qtable_path}': {exc}")
        return

    q_cols = [c for c in ["Q_HIT", "Q_STAND", "Q_DOUBLE", "Q_SPLIT", "Q_SURRENDER"]
              if c in df.columns]
    if not q_cols:
        print("  Skipping Q-heatmap: no Q_* columns found in Q-table CSV.")
        return

    # Hard totals only
    hard = df[df["usable_ace"] == 0].copy()
    if hard.empty:
        print("  Skipping Q-heatmap: no hard-total rows in Q-table.")
        return

    hard["best_q"] = hard[q_cols].max(axis=1)
    hard["best_action"] = hard[q_cols].idxmax(axis=1).str.replace("Q_", "", regex=False)

    # Grid axes
    player_totals = list(range(4, 22))   # rows 4-21
    dealer_cards  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]  # cols; 1 = Ace
    dealer_labels = [str(c) if c != 1 else "A" for c in dealer_cards]

    best_q_grid   = np.full((len(player_totals), len(dealer_cards)), np.nan)
    action_labels = [["" for _ in dealer_cards] for _ in player_totals]

    for _, row in hard.iterrows():
        pt = int(row["player_total"])
        dc = int(row["dealer_card"])
        if pt in player_totals and dc in dealer_cards:
            pi = player_totals.index(pt)
            di = dealer_cards.index(dc)
            best_q_grid[pi, di] = row["best_q"]
            action_labels[pi][di] = ACTION_CHARS.get(str(row["best_action"]), "?")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_HEATMAP)
    vmin = float(np.nanmin(best_q_grid))
    vmax = float(np.nanmax(best_q_grid))

    im = ax.imshow(best_q_grid, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    # Axis labels
    ax.set_xticks(range(len(dealer_cards)))
    ax.set_xticklabels(dealer_labels, fontsize=11)
    ax.set_yticks(range(len(player_totals)))
    ax.set_yticklabels(player_totals, fontsize=9)
    ax.set_xlabel("Dealer Up Card", fontsize=12)
    ax.set_ylabel("Player Total", fontsize=12)
    ax.set_title(
        "Q-Value Heatmap — Best Action Value per State (Hard Totals)\n"
        "H=Hit  S=Stand  D=Double  R=Surrender  (uppercase = match basic strategy is implicit)",
        fontsize=12, fontweight="bold",
    )

    # Cell text — white on dark cells, black on light cells
    for pi in range(len(player_totals)):
        for di in range(len(dealer_cards)):
            val = best_q_grid[pi, di]
            if np.isnan(val):
                continue
            # Normalise to [0,1] for brightness decision
            norm_val = (val - vmin) / (vmax - vmin + 1e-9)
            text_color = "white" if norm_val > 0.65 or norm_val < 0.35 else "black"
            label = action_labels[pi][di]
            if label:
                ax.text(di, pi, label, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    plt.colorbar(im, ax=ax, label="Best Q-Value", shrink=0.85)
    fig.tight_layout()
    path = os.path.join(output_dir, "q_heatmap.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---- main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plot_training",
        description="Visualise Blackjack Q-Learning training results",
        epilog=(
            "Examples:\n"
            "  python plot_training.py logs/training_*.csv\n"
            "  python plot_training.py logs/*.csv --output results/ --qtable analysis/q_table.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        metavar="CSV",
        help="Training log CSV file(s). Supports shell glob patterns.",
    )
    parser.add_argument(
        "--output", "-o",
        default="plots/",
        metavar="DIR",
        help="Output directory for PNG plots (default: plots/)",
    )
    parser.add_argument(
        "--qtable",
        default="analysis/q_table.csv",
        metavar="PATH",
        help="Q-table CSV for heatmap (default: analysis/q_table.csv)",
    )
    args = parser.parse_args()

    # Expand any shell glob patterns passed as strings (needed on Windows)
    expanded: list[str] = []
    for pattern in args.csv_files:
        matches = glob_mod.glob(pattern)
        expanded.extend(matches if matches else [pattern])

    print(f"Loading {len(expanded)} log file(s)...")
    data = load_logs(expanded)
    episode_min = int(data["episode"].min())
    episode_max = int(data["episode"].max())
    print(
        f"  {len(data):,} evaluation points  "
        f"(episodes {episode_min:,} – {episode_max:,})"
    )

    os.makedirs(args.output, exist_ok=True)
    print(f"\nGenerating plots in '{args.output}/'...")

    plot_learning_curve(data, args.output)
    plot_reward_curve(data, args.output)
    plot_epsilon_decay(data, args.output)
    plot_state_coverage(data, args.output)
    plot_q_heatmap(args.qtable, args.output)

    print("\nDone. Open the PNG files in your viewer of choice.")


if __name__ == "__main__":
    main()
