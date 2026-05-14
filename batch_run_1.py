# batch_run.py (v2)

"""
Batch experiment for the recommendation-backfire ABM.
This script runs a large number of model instances across a sweep of key parameters,
collects final-step metrics, and saves the results to a CSV file for later analysis.
"""

import time
import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from model import RecommendationBackfireModel


# -----------------------------
# Experiment configuration
# -----------------------------

# Main sweep: algorithmic recommendation diversity.
PREFERENCE_WIDTHS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00]

# Ablation: adaptive tolerance on or off.
ADAPTIVE_TOLERANCE_VALUES = [True, False]

# Initial ideological distribution of agents.
# polarized: two clusters around +/- 0.45 (default in model.py)
# uniform:   random across [-1, 1]
# moderate:  centered around 0 with small spread
INITIAL_DISTRIBUTIONS = ["polarized", "uniform", "moderate"]

# Independent random seeds per cell.
N_SEEDS = 20

# Fixed model parameters.
FIXED_PARAMS = {
    "num_agents": 200,
    "high_tolerance_share": 0.5,
    "feedback_sensitivity": 0.25,
    "assimilation_rate": 0.08,
    "backfire_rate": 0.06,
}

N_STEPS = 100

OUTPUT_PATH = "batch_results.csv"


# -----------------------------
# Single run
# -----------------------------

def run_one(args):
    """
    Run one model instance and return a dict of final-step metrics.
    Designed to be called inside multiprocessing.Pool.
    """

    width, adaptive, distribution, seed = args

    model = RecommendationBackfireModel(
        initial_preference_width=width,
        adaptive_tolerance=adaptive,
        initial_distribution=distribution,
        seed=seed,
        **FIXED_PARAMS,
    )

    model.run_model(N_STEPS)

    df = model.datacollector.get_model_vars_dataframe()
    final = df.iloc[-1]
    initial = df.iloc[0]

    return {
        # Sweep parameters
        "initial_preference_width": width,
        "adaptive_tolerance": adaptive,
        "initial_distribution": distribution,
        "seed": seed,
        # Initial-step values (so we can measure change, not just endpoint)
        "initial_mean_extremity": initial["Mean Extremity"],
        "initial_opinion_variance": initial["Opinion Variance"],
        "initial_extreme_share": initial["Extreme Share"],
        # Final-step outcomes
        "final_mean_opinion": final["Mean Opinion"],
        "final_mean_extremity": final["Mean Extremity"],
        "final_opinion_variance": final["Opinion Variance"],
        "final_extreme_share": final["Extreme Share"],
        "final_preference_width": final["Mean Preference Width"],
        "final_preference_extremity": final["Mean Preference Center Extremity"],
        "final_acceptance_threshold": final["Mean Acceptance Threshold"],
        "final_rejection_threshold": final["Mean Rejection Threshold"],
        "final_acceptance_rate": final["Acceptance Rate"],
        "final_ignore_rate": final["Ignore Rate"],
        "final_backfire_rate": final["Backfire Rate"],
        "final_avg_exposure_distance": final["Average Exposure Distance"],
    }


# -----------------------------
# Main
# -----------------------------

def build_arg_list():
    """
    Build the full list of (width, adaptive, distribution, seed) combinations.
    """

    seeds = list(range(1, N_SEEDS + 1))

    return list(itertools.product(
        PREFERENCE_WIDTHS,
        ADAPTIVE_TOLERANCE_VALUES,
        INITIAL_DISTRIBUTIONS,
        seeds,
    ))


def main():
    args_list = build_arg_list()
    n_runs = len(args_list)

    n_workers = max(1, cpu_count() - 1)

    print("=" * 60)
    print("Recommendation-Backfire ABM: Batch Experiment v2")
    print("=" * 60)
    print(f"Preference widths:     {PREFERENCE_WIDTHS}")
    print(f"Adaptive tolerance:    {ADAPTIVE_TOLERANCE_VALUES}")
    print(f"Initial distributions: {INITIAL_DISTRIBUTIONS}")
    print(f"Seeds per cell:        {N_SEEDS}")
    print(f"Total runs:            {n_runs}")
    print(f"Steps per run:         {N_STEPS}")
    print(f"Workers:               {n_workers}")
    print("=" * 60)

    start = time.time()

    with Pool(processes=n_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(run_one, args_list), start=1):
            results.append(result)
            if i % 50 == 0 or i == n_runs:
                elapsed = time.time() - start
                rate = i / elapsed
                remaining = (n_runs - i) / rate if rate > 0 else 0
                print(
                    f"  Completed {i}/{n_runs} runs "
                    f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)"
                )
    df = pd.DataFrame(results)

    df = df.sort_values(
        ["initial_distribution", "adaptive_tolerance", "initial_preference_width", "seed"]
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    total_time = time.time() - start
    print("=" * 60)
    print(f"Done. {n_runs} runs in {total_time:.1f}s "
          f"({total_time / n_runs:.2f}s/run avg).")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("Next: run `python batch_analysis.py` to produce figures.")
    print("=" * 60)


if __name__ == "__main__":
    main()
