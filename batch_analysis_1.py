# batch_analysis.py (v2)

"""
Analysis and visualization for batch_results.csv (v2).

Reads the 3D sweep (width x adaptive x distribution) and produces:

    fig1_main_extremity_by_dist.png   - 3-panel: extremity vs width, one panel per distribution
    fig2_extreme_share_by_dist.png    - 3-panel: extreme share vs width
    fig3_delta_extremity.png          - 3-panel: change in extremity (final - initial)
    fig4_backfire_rate_by_dist.png    - 3-panel: backfire rate vs width
    fig5_threshold_drift_by_dist.png  - 3-panel: adaptive tolerance evidence
    fig6_summary_grid.png             - 4-panel: main outcomes for the polarized condition
    fig7_essay_main.png               - the single most important figure for the paper

Also prints a numeric summary table and a non-monotonicity check.

Run with:
    python batch_analysis_1.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "batch_results.csv"

DISTRIBUTIONS = ["polarized", "uniform", "moderate"]
ADAPTIVE_COLORS = {True: "#d62728", False: "#1f77b4"}
ADAPTIVE_LABELS = {True: "Adaptive ON", False: "Adaptive OFF"}


# -----------------------------
# Helpers
# -----------------------------

def summarize(df, group_cols, value_col):
    """Mean and standard deviation of value_col, grouped by group_cols."""
    g = df.groupby(group_cols)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": "mean_val", "std": "std_val", "count": "n"})
    return out


def plot_with_band(ax, x, mean, std, label, color):
    """Line plot with a +/-1 SD shaded band."""
    ax.plot(x, mean, marker="o", linewidth=2, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)


def faceted_by_distribution(df, value_col, title, y_label, filename, ylim=None):
    """
    3-panel plot, one panel per initial_distribution.
    Each panel shows two lines (adaptive on/off) over width.
    """

    summary = summarize(
        df,
        ["initial_distribution", "adaptive_tolerance", "initial_preference_width"],
        value_col,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, dist in zip(axes, DISTRIBUTIONS):
        for adaptive_val in [True, False]:
            sub = summary[
                (summary["initial_distribution"] == dist)
                & (summary["adaptive_tolerance"] == adaptive_val)
            ].sort_values("initial_preference_width")

            plot_with_band(
                ax,
                sub["initial_preference_width"].values,
                sub["mean_val"].values,
                sub["std_val"].values,
                label=ADAPTIVE_LABELS[adaptive_val],
                color=ADAPTIVE_COLORS[adaptive_val],
            )

        ax.set_title(f"Initial: {dist}")
        ax.set_xlabel("Initial recommendation width")
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(y_label)
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def faceted_delta(df, final_col, initial_col, title, y_label, filename):
    """
    3-panel plot of CHANGE: final_col - initial_col.

    Critical for distinguishing "frozen" outcomes (low extremity because
    nothing happened) from "true depolarization" outcomes (low extremity
    because system actively moderated).
    """

    df = df.copy()
    df["__delta"] = df[final_col] - df[initial_col]

    summary = summarize(
        df,
        ["initial_distribution", "adaptive_tolerance", "initial_preference_width"],
        "__delta",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, dist in zip(axes, DISTRIBUTIONS):
        for adaptive_val in [True, False]:
            sub = summary[
                (summary["initial_distribution"] == dist)
                & (summary["adaptive_tolerance"] == adaptive_val)
            ].sort_values("initial_preference_width")

            plot_with_band(
                ax,
                sub["initial_preference_width"].values,
                sub["mean_val"].values,
                sub["std_val"].values,
                label=ADAPTIVE_LABELS[adaptive_val],
                color=ADAPTIVE_COLORS[adaptive_val],
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"Initial: {dist}")
        ax.set_xlabel("Initial recommendation width")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(y_label)
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


# -----------------------------
# Specialized figures
# -----------------------------

def summary_grid_polarized(df, filename):
    """
    Four-panel figure for the polarized condition only.
    Reproduces the original v1 figure for direct comparison.
    """

    sub_df = df[df["initial_distribution"] == "polarized"]

    metrics = [
        ("final_mean_extremity", "Mean ideological extremity",
         "(A) Polarization vs recommendation diversity"),
        ("final_extreme_share", "Share of agents at poles (|x| >= 0.75)",
         "(B) Extreme share vs recommendation diversity"),
        ("final_preference_width", "Final algorithmic recommendation width",
         "(C) Algorithmic narrowing"),
        ("final_backfire_rate", "Cumulative rejection rate",
         "(D) Backfire rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (col, ylabel, title) in zip(axes, metrics):
        summary = summarize(
            sub_df,
            ["adaptive_tolerance", "initial_preference_width"],
            col,
        )

        for adaptive_val in [True, False]:
            sub = summary[summary["adaptive_tolerance"] == adaptive_val].sort_values(
                "initial_preference_width"
            )
            plot_with_band(
                ax,
                sub["initial_preference_width"].values,
                sub["mean_val"].values,
                sub["std_val"].values,
                label=ADAPTIVE_LABELS[adaptive_val],
                color=ADAPTIVE_COLORS[adaptive_val],
            )

        ax.set_xlabel("Initial recommendation width")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Polarized Initial Condition: Diversity, Backfire, and Polarization",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def essay_main_figure(df, filename):
    """
    The single most informative figure for the essay.

    Layout:
        Top row: final extremity by width, faceted by distribution.
        Bottom row: change in extremity (final - initial), faceted by distribution.

    The bottom row is the cleanest evidence on whether the algorithm DRIVES
    polarization (positive delta) or not (zero or negative delta).
    """

    df = df.copy()
    df["__delta_extremity"] = df["final_mean_extremity"] - df["initial_mean_extremity"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)

    # Top row: final extremity
    top_summary = summarize(
        df,
        ["initial_distribution", "adaptive_tolerance", "initial_preference_width"],
        "final_mean_extremity",
    )

    for ax, dist in zip(axes[0], DISTRIBUTIONS):
        for adaptive_val in [True, False]:
            sub = top_summary[
                (top_summary["initial_distribution"] == dist)
                & (top_summary["adaptive_tolerance"] == adaptive_val)
            ].sort_values("initial_preference_width")
            plot_with_band(
                ax,
                sub["initial_preference_width"].values,
                sub["mean_val"].values,
                sub["std_val"].values,
                label=ADAPTIVE_LABELS[adaptive_val],
                color=ADAPTIVE_COLORS[adaptive_val],
            )
        ax.set_title(f"Initial: {dist}")
        ax.grid(True, alpha=0.3)

    axes[0][0].set_ylabel("Final mean extremity")
    axes[0][0].legend(loc="best", fontsize=9)

    # Bottom row: change in extremity
    bot_summary = summarize(
        df,
        ["initial_distribution", "adaptive_tolerance", "initial_preference_width"],
        "__delta_extremity",
    )

    for ax, dist in zip(axes[1], DISTRIBUTIONS):
        for adaptive_val in [True, False]:
            sub = bot_summary[
                (bot_summary["initial_distribution"] == dist)
                & (bot_summary["adaptive_tolerance"] == adaptive_val)
            ].sort_values("initial_preference_width")
            plot_with_band(
                ax,
                sub["initial_preference_width"].values,
                sub["mean_val"].values,
                sub["std_val"].values,
                label=ADAPTIVE_LABELS[adaptive_val],
                color=ADAPTIVE_COLORS[adaptive_val],
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Initial recommendation width")
        ax.grid(True, alpha=0.3)

    axes[1][0].set_ylabel("Change in extremity (final - initial)")

    fig.suptitle(
        "Recommendation Diversity Drives Polarization Across Initial Conditions",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


# -----------------------------
# Numerical summaries
# -----------------------------

def print_summary_table(df):
    """Print a tidy table of key outcomes by (distribution, adaptive, width)."""

    summary = (
        df.groupby([
            "initial_distribution", "adaptive_tolerance", "initial_preference_width"
        ])
        .agg(
            mean_extremity=("final_mean_extremity", "mean"),
            sd_extremity=("final_mean_extremity", "std"),
            extreme_share=("final_extreme_share", "mean"),
            backfire_rate=("final_backfire_rate", "mean"),
            final_width=("final_preference_width", "mean"),
        )
        .round(3)
        .reset_index()
    )

    print("\nOutcome means by cell (averaged over seeds):")
    for dist in DISTRIBUTIONS:
        sub = summary[summary["initial_distribution"] == dist]
        print(f"\n--- initial_distribution = {dist} ---")
        print(sub.to_string(index=False))


def print_emergent_polarization_check(df):
    """
    Critical check for the essay claim:

    Under the UNIFORM initial distribution (no prior polarization), does
    the system MOVE TOWARD polarization as recommendation width increases?

    A positive delta_extremity here means the algorithm is actively creating
    polarization, not just amplifying existing polarization.
    """

    print("\nEmergent polarization check (uniform initial condition):")
    print("Positive delta = algorithm actively created polarization.\n")

    sub_df = df[df["initial_distribution"] == "uniform"].copy()
    sub_df["__delta"] = sub_df["final_mean_extremity"] - sub_df["initial_mean_extremity"]

    g = sub_df.groupby(["adaptive_tolerance", "initial_preference_width"])["__delta"]
    summary = g.mean().round(3).reset_index()

    for adaptive_val in [True, False]:
        sub = summary[summary["adaptive_tolerance"] == adaptive_val].sort_values(
            "initial_preference_width"
        )
        print(f"  adaptive={adaptive_val}:")
        for _, row in sub.iterrows():
            sign = "+" if row["__delta"] >= 0 else ""
            print(
                f"    width={row['initial_preference_width']:.2f}  "
                f"delta_extremity = {sign}{row['__delta']:.3f}"
            )


def print_moderate_breakdown_check(df):
    """
    Under the MODERATE initial distribution, does wide recommendation
    BREAK moderation? A positive delta means the algorithm pushed a
    moderate population toward extremes.
    """

    print("\nModerate-breakdown check (moderate initial condition):")
    print("Positive delta = algorithm pushed a moderate population apart.\n")

    sub_df = df[df["initial_distribution"] == "moderate"].copy()
    sub_df["__delta"] = sub_df["final_mean_extremity"] - sub_df["initial_mean_extremity"]

    g = sub_df.groupby(["adaptive_tolerance", "initial_preference_width"])["__delta"]
    summary = g.mean().round(3).reset_index()

    for adaptive_val in [True, False]:
        sub = summary[summary["adaptive_tolerance"] == adaptive_val].sort_values(
            "initial_preference_width"
        )
        print(f"  adaptive={adaptive_val}:")
        for _, row in sub.iterrows():
            sign = "+" if row["__delta"] >= 0 else ""
            print(
                f"    width={row['initial_preference_width']:.2f}  "
                f"delta_extremity = {sign}{row['__delta']:.3f}"
            )


def print_adaptive_buffer_check(df):
    """
    Compare adaptive ON vs OFF: at the widest recommendation setting,
    does adaptive tolerance dampen the polarization that wide exposure
    creates? Reports the gap (off - on) for each distribution.

    Positive gap = adaptive tolerance acts as a buffer against algorithmic
    polarization. This is the mirror image of "candidate 3" (adaptive
    amplifies) and is the actual finding pattern from v1 results.
    """

    print("\nAdaptive-buffer check (adaptive OFF minus adaptive ON):")
    print("Positive gap = adaptive tolerance reduced polarization.\n")

    g = (
        df.groupby([
            "initial_distribution", "adaptive_tolerance", "initial_preference_width"
        ])["final_mean_extremity"].mean().reset_index()
    )

    for dist in DISTRIBUTIONS:
        print(f"  initial_distribution = {dist}:")
        sub = g[g["initial_distribution"] == dist]
        on = sub[sub["adaptive_tolerance"] == True].set_index("initial_preference_width")["final_mean_extremity"]
        off = sub[sub["adaptive_tolerance"] == False].set_index("initial_preference_width")["final_mean_extremity"]
        gap = (off - on).round(3)
        for w, v in gap.items():
            sign = "+" if v >= 0 else ""
            print(f"    width={w:.2f}  off-on = {sign}{v:.3f}")


# -----------------------------
# Main
# -----------------------------

def main():
    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print(f"Loaded {len(df)} runs from {INPUT_PATH}")
    print(f"Distributions: {sorted(df['initial_distribution'].unique())}")
    print(f"Widths:        {sorted(df['initial_preference_width'].unique())}")
    print(f"Adaptive vals: {sorted(df['adaptive_tolerance'].unique())}")
    print(f"Seeds/cell:    "
          f"{df.groupby(['initial_distribution', 'adaptive_tolerance', 'initial_preference_width']).size().min()}")
    print("=" * 60)

    print("\nGenerating figures...")

    faceted_by_distribution(
        df,
        value_col="final_mean_extremity",
        title="Final Mean Extremity vs Recommendation Width",
        y_label="Final mean ideological extremity",
        filename="fig1_main_extremity_by_dist.png",
    )

    faceted_by_distribution(
        df,
        value_col="final_extreme_share",
        title="Share of Agents at Poles vs Recommendation Width",
        y_label="Final extreme share (|opinion| >= 0.75)",
        filename="fig2_extreme_share_by_dist.png",
    )

    faceted_delta(
        df,
        final_col="final_mean_extremity",
        initial_col="initial_mean_extremity",
        title="Change in Mean Extremity (final - initial)",
        y_label="Change in mean extremity",
        filename="fig3_delta_extremity.png",
    )

    faceted_by_distribution(
        df,
        value_col="final_backfire_rate",
        title="Cumulative Backfire Rate vs Recommendation Width",
        y_label="Share of recommendations rejected",
        filename="fig4_backfire_rate_by_dist.png",
    )

    faceted_by_distribution(
        df,
        value_col="final_acceptance_threshold",
        title="Final Acceptance Threshold (adaptive tolerance evidence)",
        y_label="Final mean acceptance threshold",
        filename="fig5_threshold_drift_by_dist.png",
    )

    summary_grid_polarized(df, filename="fig6_summary_grid_polarized.png")

    essay_main_figure(df, filename="fig7_essay_main.png")

    # Numerical summaries
    print_summary_table(df)
    print_emergent_polarization_check(df)
    print_moderate_breakdown_check(df)
    print_adaptive_buffer_check(df)

    print("\nDone. Open the PNG files to inspect the results.")


if __name__ == "__main__":
    main()
