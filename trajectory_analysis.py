# trajectory_analysis.py

"""
Experiment C: Trajectory analysis.

Instead of comparing only step-100 endpoints, this script collects the full
step-by-step trajectory of polarization for a selected subset of conditions.

Why this matters for the paper:
- It distinguishes systems that polarize quickly and stabilize from systems
  that keep drifting throughout the simulation window.
- It lets us see whether 100 steps is enough for the system to reach a
  steady state, or whether more polarization would emerge with longer runs.
- It produces a dynamic figure that complements the static endpoint plots.

This script does NOT modify model.py or agents.py. It just calls the model
with the same parameters and reads the per-step DataCollector dataframe.

Run with:
    python trajectory_analysis.py
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import RecommendationBackfireModel


# -----------------------------
# Configuration
# -----------------------------

# A small selected set of conditions to plot trajectories for.
# We pick three widths: narrow (0.2), medium (0.5), wide (1.0).
# We use the polarized initial distribution and adaptive=False as the cleanest
# baseline. (Adding more conditions would clutter the plot.)
WIDTHS = [0.20, 0.50, 1.00]
DISTRIBUTIONS = ["polarized", "uniform", "moderate"]
ADAPTIVE = False

N_SEEDS = 10  # fewer seeds because we plot trajectories, not endpoints
N_STEPS = 200  # longer than the main batch, to check whether 100 was enough

FIXED_PARAMS = {
    "num_agents": 200,
    "high_tolerance_share": 0.5,
    "feedback_sensitivity": 0.25,
    "initial_preference_width": None,  # set per run
    "assimilation_rate": 0.08,
    "backfire_rate": 0.06,
}


# -----------------------------
# Run
# -----------------------------

def collect_trajectory(width, distribution, seed):
    """
    Run one model and return its per-step model-level dataframe with
    bookkeeping columns added.
    """

    params = dict(FIXED_PARAMS)
    params["initial_preference_width"] = width

    model = RecommendationBackfireModel(
        initial_distribution=distribution,
        adaptive_tolerance=ADAPTIVE,
        seed=seed,
        **params,
    )

    model.run_model(N_STEPS)

    df = model.datacollector.get_model_vars_dataframe().reset_index()
    df = df.rename(columns={"index": "step"})
    df["initial_preference_width"] = width
    df["initial_distribution"] = distribution
    df["seed"] = seed
    return df


def main():
    start = time.time()
    rows = []

    n_runs = len(WIDTHS) * len(DISTRIBUTIONS) * N_SEEDS
    print(f"Running {n_runs} trajectories x {N_STEPS} steps each...")

    i = 0
    for dist in DISTRIBUTIONS:
        for width in WIDTHS:
            for seed in range(1, N_SEEDS + 1):
                rows.append(collect_trajectory(width, dist, seed))
                i += 1
                if i % 10 == 0:
                    print(f"  {i}/{n_runs} done")

    df = pd.concat(rows, ignore_index=True)
    df.to_csv("trajectory_results.csv", index=False)
    print(f"Saved trajectory_results.csv ({len(df)} rows) in {time.time()-start:.1f}s")

    # -----------------------------
    # Plot: 3-panel by distribution; one line per width with seed-band.
    # -----------------------------

    summary = (
        df.groupby(["initial_distribution", "initial_preference_width", "step"])
        .agg(mean_extremity=("Mean Extremity", "mean"),
             sd_extremity=("Mean Extremity", "std"))
        .reset_index()
    )

    width_colors = {0.20: "#2ca02c", 0.50: "#ff7f0e", 1.00: "#d62728"}
    width_labels = {0.20: "Narrow (w=0.20)",
                    0.50: "Medium (w=0.50)",
                    1.00: "Wide   (w=1.00)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, dist in zip(axes, DISTRIBUTIONS):
        sub = summary[summary["initial_distribution"] == dist]
        for w in WIDTHS:
            ssub = sub[sub["initial_preference_width"] == w].sort_values("step")
            x = ssub["step"].values
            m = ssub["mean_extremity"].values
            s = ssub["sd_extremity"].values
            ax.plot(x, m, linewidth=2, color=width_colors[w], label=width_labels[w])
            ax.fill_between(x, m - s, m + s, alpha=0.2, color=width_colors[w])

        ax.axvline(100, color="black", linestyle=":", alpha=0.5,
                   label="end of main batch (step 100)")
        ax.set_title(f"Initial: {dist}")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean ideological extremity")
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle("Polarization Trajectories Over Time (adaptive tolerance OFF)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("fig_trajectories.png", dpi=150)
    plt.close(fig)
    print("Saved fig_trajectories.png")

    # -----------------------------
    # Quick numeric check: did the wide condition still drift between 100-200?
    # -----------------------------

    print("\nDrift check (Mean Extremity at step 100 vs step 199):")
    for dist in DISTRIBUTIONS:
        for w in WIDTHS:
            sub = df[(df["initial_distribution"] == dist)
                     & (df["initial_preference_width"] == w)]
            at_100 = sub[sub["step"] == 100]["Mean Extremity"].mean()
            at_199 = sub[sub["step"] == 199]["Mean Extremity"].mean()
            print(f"  {dist:10s} w={w:.2f}:  "
                  f"step 100 = {at_100:.3f}   step 199 = {at_199:.3f}   "
                  f"drift = {at_199-at_100:+.3f}")


if __name__ == "__main__":
    main()
