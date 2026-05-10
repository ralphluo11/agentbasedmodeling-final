# trajectory_analysis_v2.py

"""
Experiment C v2: Long-horizon trajectory analysis for Finding 4 (overshoot).

Changes from v1:
    - 400 steps instead of 200 (test whether the relaxation continues
      beyond the original window).
    - 20 seeds instead of 10 (cleaner SD bands; matches main batch).
    - Parallelized (180 runs * 400 steps takes ~6-10 min serial; faster in parallel).
    - Three numeric diagnostics added:
        (a) peak step and peak value per condition,
        (b) magnitude of relaxation from peak to step 400,
        (c) late-stage drift (slope between step 300 and step 400)
            -- tells us whether the system is still drifting or has stabilized.

The goal is to characterize the overshoot pattern precisely enough to
support strong claims in Finding 4 of the essay.

Run with:
    python trajectory_analysis_v2.py
"""

import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import RecommendationBackfireModel


# -----------------------------
# Configuration
# -----------------------------

WIDTHS = [0.20, 0.50, 1.00]
DISTRIBUTIONS = ["polarized", "uniform", "moderate"]
ADAPTIVE = False

N_SEEDS = 20
N_STEPS = 400

FIXED_PARAMS = {
    "num_agents": 200,
    "high_tolerance_share": 0.5,
    "feedback_sensitivity": 0.25,
    "assimilation_rate": 0.08,
    "backfire_rate": 0.06,
}


# -----------------------------
# Single run
# -----------------------------

def run_one(args):
    """Run one model, return per-step dataframe with bookkeeping columns."""

    width, distribution, seed = args

    model = RecommendationBackfireModel(
        initial_preference_width=width,
        adaptive_tolerance=ADAPTIVE,
        initial_distribution=distribution,
        seed=seed,
        **FIXED_PARAMS,
    )
    model.run_model(N_STEPS)

    df = model.datacollector.get_model_vars_dataframe().reset_index()
    df = df.rename(columns={"index": "step"})
    df["initial_preference_width"] = width
    df["initial_distribution"] = distribution
    df["seed"] = seed
    return df


# -----------------------------
# Diagnostics
# -----------------------------

def compute_diagnostics(df):
    """
    For each (distribution, width) cell, compute:
        peak_step      : step at which mean extremity peaks (averaged over seeds)
        peak_value     : peak mean extremity
        end_value      : mean extremity at the final step
        relaxation     : peak_value - end_value  (positive = overshoot)
        late_slope     : (end_value - value at 75% of run) / (steps in last 25%)
                         positive = still polarizing, negative = still relaxing,
                         near zero = stabilized.
    """

    diagnostics = []

    for dist in DISTRIBUTIONS:
        for w in WIDTHS:
            sub = df[(df["initial_distribution"] == dist)
                     & (df["initial_preference_width"] == w)]
            avg = sub.groupby("step")["Mean Extremity"].mean()

            peak_step = int(avg.idxmax())
            peak_value = float(avg.max())
            end_value = float(avg.iloc[-1])
            relaxation = peak_value - end_value

            quarter_step = int(N_STEPS * 0.75)
            mid_value = float(avg.iloc[quarter_step])
            late_slope_per_100 = (
                (end_value - mid_value) / (N_STEPS - quarter_step) * 100
            )

            diagnostics.append({
                "distribution": dist,
                "width": w,
                "peak_step": peak_step,
                "peak_value": round(peak_value, 4),
                "end_value": round(end_value, 4),
                "relaxation": round(relaxation, 4),
                "late_slope_per_100_steps": round(late_slope_per_100, 4),
            })

    return pd.DataFrame(diagnostics)


# -----------------------------
# Plotting
# -----------------------------

def plot_trajectories(df, filename):
    """3-panel by distribution; one line per width with seed-band."""

    summary = (
        df.groupby(["initial_distribution", "initial_preference_width", "step"])
        .agg(mean=("Mean Extremity", "mean"),
             sd=("Mean Extremity", "std"))
        .reset_index()
    )

    width_colors = {0.20: "#2ca02c", 0.50: "#ff7f0e", 1.00: "#d62728"}
    width_labels = {0.20: "Narrow (w=0.20)",
                    0.50: "Medium (w=0.50)",
                    1.00: "Wide   (w=1.00)"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, dist in zip(axes, DISTRIBUTIONS):
        sub = summary[summary["initial_distribution"] == dist]
        for w in WIDTHS:
            ssub = sub[sub["initial_preference_width"] == w].sort_values("step")
            x = ssub["step"].values
            m = ssub["mean"].values
            s = ssub["sd"].values
            ax.plot(x, m, linewidth=2, color=width_colors[w], label=width_labels[w])
            ax.fill_between(x, m - s, m + s, alpha=0.2, color=width_colors[w])

            # Mark the peak with a vertical dashed line for the wide condition.
            if w == 1.00:
                peak_step = int(np.argmax(m))
                ax.axvline(peak_step, color=width_colors[w], linestyle=":",
                           alpha=0.5, linewidth=1)

        # Mark the original step-100 endpoint of the main batch for reference.
        ax.axvline(100, color="black", linestyle=":", alpha=0.4, linewidth=1,
                   label="end of main batch (step 100)")
        ax.set_title(f"Initial: {dist}")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean ideological extremity")
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle(
        "Polarization Trajectories Over 400 Steps (adaptive tolerance OFF)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_relaxation_focus(df, filename):
    """
    Zoomed view of the wide (w=1.00) condition only, all three distributions
    overlaid. Makes the overshoot/relaxation pattern very clear.
    """

    sub = df[df["initial_preference_width"] == 1.00]
    summary = (
        sub.groupby(["initial_distribution", "step"])
        .agg(mean=("Mean Extremity", "mean"),
             sd=("Mean Extremity", "std"))
        .reset_index()
    )

    dist_colors = {"polarized": "#d62728", "uniform": "#1f77b4", "moderate": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for dist in DISTRIBUTIONS:
        ssub = summary[summary["initial_distribution"] == dist].sort_values("step")
        x = ssub["step"].values
        m = ssub["mean"].values
        s = ssub["sd"].values
        ax.plot(x, m, linewidth=2, color=dist_colors[dist],
                label=f"Initial: {dist}")
        ax.fill_between(x, m - s, m + s, alpha=0.2, color=dist_colors[dist])

    ax.axvline(100, color="black", linestyle=":", alpha=0.4, linewidth=1,
               label="step 100 (main batch endpoint)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean ideological extremity")
    ax.set_title("Overshoot-Relaxation Pattern at Wide Recommendation (w=1.00)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


# -----------------------------
# Main
# -----------------------------

def main():
    args_list = [
        (w, d, s)
        for d in DISTRIBUTIONS
        for w in WIDTHS
        for s in range(1, N_SEEDS + 1)
    ]
    n_runs = len(args_list)
    n_workers = max(1, cpu_count() - 1)

    print("=" * 60)
    print("Experiment C v2: Long-Horizon Trajectory Analysis")
    print("=" * 60)
    print(f"Widths:        {WIDTHS}")
    print(f"Distributions: {DISTRIBUTIONS}")
    print(f"Seeds/cell:    {N_SEEDS}")
    print(f"Steps per run: {N_STEPS}")
    print(f"Total runs:    {n_runs}")
    print(f"Workers:       {n_workers}")
    print("=" * 60)

    start = time.time()

    with Pool(processes=n_workers) as pool:
        rows = []
        for i, df in enumerate(pool.imap_unordered(run_one, args_list), start=1):
            rows.append(df)
            if i % 20 == 0 or i == n_runs:
                elapsed = time.time() - start
                rate = i / elapsed
                remaining = (n_runs - i) / rate if rate > 0 else 0
                print(f"  {i}/{n_runs} runs ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")

    # Serial fallback (uncomment if multiprocessing fails on Windows):
    # rows = [run_one(a) for a in args_list]

    df = pd.concat(rows, ignore_index=True)
    df.to_csv("trajectory_results_v2.csv", index=False)
    print(f"\nSaved trajectory_results_v2.csv ({len(df)} rows)")

    # -----------------------------
    # Plots
    # -----------------------------
    print("\nGenerating figures...")
    plot_trajectories(df, "fig_trajectories_v2.png")
    plot_relaxation_focus(df, "fig_overshoot_focus.png")

    # -----------------------------
    # Diagnostics
    # -----------------------------
    diag = compute_diagnostics(df)
    diag.to_csv("trajectory_diagnostics.csv", index=False)

    print("\n" + "=" * 60)
    print("OVERSHOOT DIAGNOSTICS")
    print("=" * 60)
    print(diag.to_string(index=False))

    print("\nInterpretation guide:")
    print("  peak_step:    when does mean extremity peak (within 0..N_STEPS)?")
    print("                If close to N_STEPS, the system has not yet peaked.")
    print("  relaxation:   peak_value - end_value")
    print("                > 0.02 = clear overshoot pattern")
    print("                ~ 0    = monotonic plateau")
    print("                < 0    = bug (peak occurred at end)")
    print("  late_slope_per_100_steps: change in extremity between step 300 and step 400")
    print("                |slope| < 0.01 = stabilized")
    print("                slope > +0.01  = still polarizing (need longer run)")
    print("                slope < -0.01  = still relaxing (need longer run)")

    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
