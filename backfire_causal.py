# backfire_causal.py

"""
Experiment A: Causal test of the backfire mechanism.
"""

import time
import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import RecommendationBackfireModel


# -----------------------------
# Configuration
# -----------------------------

BACKFIRE_RATES = [0.00, 0.03, 0.06, 0.12]
WIDTHS = [0.20, 0.40, 0.60, 0.85, 1.00]
ADAPTIVE_VALUES = [True, False]
N_SEEDS = 20

INITIAL_DISTRIBUTION = "uniform"
N_STEPS = 100

FIXED_PARAMS = {
    "num_agents": 200,
    "high_tolerance_share": 0.5,
    "feedback_sensitivity": 0.25,
    "assimilation_rate": 0.08,
}

OUTPUT_PATH = "backfire_causal_results.csv"


# -----------------------------
# Single run
# -----------------------------

def run_one(args):
    width, adaptive, br, seed = args
    model = RecommendationBackfireModel(
        initial_preference_width=width,
        adaptive_tolerance=adaptive,
        backfire_rate=br,
        initial_distribution=INITIAL_DISTRIBUTION,
        seed=seed,
        **FIXED_PARAMS,
    )
    model.run_model(N_STEPS)

    df = model.datacollector.get_model_vars_dataframe()
    initial = df.iloc[0]
    final = df.iloc[-1]

    return {
        "initial_preference_width": width,
        "adaptive_tolerance": adaptive,
        "backfire_rate": br,
        "seed": seed,
        "initial_mean_extremity": initial["Mean Extremity"],
        "final_mean_extremity": final["Mean Extremity"],
        "delta_extremity": final["Mean Extremity"] - initial["Mean Extremity"],
        "final_extreme_share": final["Extreme Share"],
        "final_opinion_variance": final["Opinion Variance"],
        "final_backfire_rate_observed": final["Backfire Rate"],
        "final_acceptance_rate": final["Acceptance Rate"],
    }


# -----------------------------
# Main
# -----------------------------

def main():
    args_list = list(itertools.product(
        WIDTHS, ADAPTIVE_VALUES, BACKFIRE_RATES, range(1, N_SEEDS + 1)
    ))
    n_runs = len(args_list)
    n_workers = max(1, cpu_count() - 1)

    print("=" * 60)
    print("Experiment A: Backfire Causal Test")
    print("=" * 60)
    print(f"backfire_rate values:   {BACKFIRE_RATES}")
    print(f"preference_width values:{WIDTHS}")
    print(f"adaptive values:        {ADAPTIVE_VALUES}")
    print(f"seeds per cell:         {N_SEEDS}")
    print(f"initial_distribution:   {INITIAL_DISTRIBUTION} (held fixed)")
    print(f"total runs:             {n_runs}")
    print(f"workers:                {n_workers}")
    print("=" * 60)

    start = time.time()
    with Pool(processes=n_workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_one, args_list), start=1):
            results.append(r)
            if i % 50 == 0 or i == n_runs:
                elapsed = time.time() - start
                rate = i / elapsed
                remaining = (n_runs - i) / rate if rate > 0 else 0
                print(f"  {i}/{n_runs} runs ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")



    df = pd.DataFrame(results).sort_values(
        ["adaptive_tolerance", "backfire_rate", "initial_preference_width", "seed"]
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH}")

    # -----------------------------
    # Numerical summary for key result: delta_extremity by (backfire_rate, width)
    # -----------------------------
    print("\n=== KEY RESULT TABLE: delta_extremity by (backfire_rate, width) ===")
    print("Adaptive OFF (cleanest test):\n")
    sub = df[df["adaptive_tolerance"] == False]
    g = sub.groupby(["backfire_rate", "initial_preference_width"])["delta_extremity"].mean().round(3)
    print(g.unstack("initial_preference_width"))
    print("\nAdaptive ON:\n")
    sub = df[df["adaptive_tolerance"] == True]
    g = sub.groupby(["backfire_rate", "initial_preference_width"])["delta_extremity"].mean().round(3)
    print(g.unstack("initial_preference_width"))

    # Falsification test: at the widest width, does delta drop substantially when br=0?
    print("\n=== FALSIFICATION TEST ===")
    print("Compare delta at br=0.0 vs br=0.06 (paper baseline) at width=1.0:")
    for adaptive in [False, True]:
        zero = df[(df["adaptive_tolerance"] == adaptive)
                  & (df["backfire_rate"] == 0.00)
                  & (df["initial_preference_width"] == 1.0)]["delta_extremity"].mean()
        baseline = df[(df["adaptive_tolerance"] == adaptive)
                      & (df["backfire_rate"] == 0.06)
                      & (df["initial_preference_width"] == 1.0)]["delta_extremity"].mean()
        attribution = baseline - zero
        print(f"  adaptive={adaptive}:  br=0 -> {zero:+.3f}   br=0.06 -> {baseline:+.3f}   "
              f"attributable to backfire: {attribution:+.3f}  "
              f"({100 * attribution / baseline:+.1f}% of baseline polarization)")

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    br_colors = {0.00: "#1f77b4", 0.03: "#2ca02c", 0.06: "#ff7f0e", 0.12: "#d62728"}

    for ax, adaptive in zip(axes, [False, True]):
        sub = df[df["adaptive_tolerance"] == adaptive]
        summary = (
            sub.groupby(["backfire_rate", "initial_preference_width"])
            .agg(mean=("delta_extremity", "mean"),
                 sd=("delta_extremity", "std"))
            .reset_index()
        )
        for br in BACKFIRE_RATES:
            s = summary[summary["backfire_rate"] == br].sort_values("initial_preference_width")
            x = s["initial_preference_width"].values
            m = s["mean"].values
            sd = s["sd"].values
            label = f"backfire_rate={br}"
            if br == 0.00:
                label += "  (no rejection push)"
            ax.plot(x, m, marker="o", linewidth=2, color=br_colors[br], label=label)
            ax.fill_between(x, m - sd, m + sd, alpha=0.15, color=br_colors[br])

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"Adaptive tolerance {'ON' if adaptive else 'OFF'}")
        ax.set_xlabel("Initial recommendation width")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axes[0].set_ylabel("Change in mean extremity (final - initial)")
    fig.suptitle(
        "Causal Test: Backfire Strength and Emergent Polarization\n"
        f"(initial distribution = {INITIAL_DISTRIBUTION})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig("fig_backfire_causal.png", dpi=150)
    plt.close(fig)
    print("Saved fig_backfire_causal.png")

    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
