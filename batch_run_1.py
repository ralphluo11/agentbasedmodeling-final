"""
Batch experiment for the recommendation-backfire ABM 
Sweep dimensions (full grid):
    - initial_preference_width        x 3  : {0.20, 0.50, 1.00}
    - adaptive_tolerance              x 2  : {True, False}
    - initial_distribution            x 3  : {polarized, uniform, moderate}
    - (social_signal_weight, trending_weight) x 4 :
          (0.0, 0.0)  — v1 individual baseline
          (0.5, 0.0)  — pure local collaborative filtering
          (0.0, 0.5)  — pure global trending
          (0.4, 0.4)  — hybrid (realistic platform)
    - seed                            x 20
"""

import time
import itertools
from multiprocessing import Pool, cpu_count

import pandas as pd

from model import RecommendationBackfireModel


# -----------------------------
# Experiment configuration
# -----------------------------

PREFERENCE_WIDTHS = [0.20, 0.50, 1.00]
ADAPTIVE_TOLERANCE_VALUES = [True, False]
INITIAL_DISTRIBUTIONS = ["polarized", "uniform", "moderate"]

# Four collaborative-filtering regimes. Each pair is (social, trending).
SOCIAL_TRENDING_REGIMES = [
    (0.0, 0.0),    # baseline (individual learning only)
    (0.5, 0.0),    # pure local CF
    (0.0, 0.5),    # pure global trending
    (0.4, 0.4),    # hybrid (realistic platform)
]

N_SEEDS = 20

FIXED_PARAMS = {
    "num_agents": 200,
    "high_tolerance_share": 0.5,
    "feedback_sensitivity": 0.25,
    "assimilation_rate": 0.08,
    "backfire_rate": 0.06,
}

N_STEPS = 100

OUTPUT_PATH = "batch_results_v2.csv"


def regime_label(s, t):
    """Human-readable label for a (social, trending) pair."""
    if s == 0.0 and t == 0.0:
        return "individual_baseline"
    if s > 0 and t == 0.0:
        return "local_cf"
    if s == 0.0 and t > 0:
        return "global_trending"
    return "hybrid"


# -----------------------------
# Single run
# -----------------------------

def run_one(args):
    """
    Run one model instance and return a dict of summary metrics.
    """

    width, adaptive, distribution, social_weight, trending_weight, seed = args

    model = RecommendationBackfireModel(
        initial_preference_width=width,
        adaptive_tolerance=adaptive,
        initial_distribution=distribution,
        social_signal_weight=social_weight,
        trending_weight=trending_weight,
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
        "social_signal_weight": social_weight,
        "trending_weight": trending_weight,
        "regime": regime_label(social_weight, trending_weight),
        "seed": seed,
        # Initial values
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
        "final_opinion_bimodality": final["Opinion Bimodality"],
        "final_acceptance_history_filled": final["Mean Acceptance History Filled"],
    }


# -----------------------------
# Main
# -----------------------------

def build_arg_list():
    seeds = list(range(1, N_SEEDS + 1))

    args = []
    for width, adaptive, distribution, (sw, tw), seed in itertools.product(
        PREFERENCE_WIDTHS,
        ADAPTIVE_TOLERANCE_VALUES,
        INITIAL_DISTRIBUTIONS,
        SOCIAL_TRENDING_REGIMES,
        seeds,
    ):
        args.append((width, adaptive, distribution, sw, tw, seed))
    return args


def main():
    args_list = build_arg_list()
    n_runs = len(args_list)

    n_workers = max(1, cpu_count() - 1)

    print("=" * 70)
    print("Recommendation-Backfire ABM: Batch Experiment v2")
    print("  (with collaborative filtering + global trending channels)")
    print("=" * 70)
    print(f"Preference widths:     {PREFERENCE_WIDTHS}")
    print(f"Adaptive tolerance:    {ADAPTIVE_TOLERANCE_VALUES}")
    print(f"Initial distributions: {INITIAL_DISTRIBUTIONS}")
    print(f"(social, trending):    {SOCIAL_TRENDING_REGIMES}")
    print(f"Seeds per cell:        {N_SEEDS}")
    print(f"Total runs:            {n_runs}")
    print(f"Steps per run:         {N_STEPS}")
    print(f"Workers:               {n_workers}")
    print("=" * 70)

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
                    f"  {i}/{n_runs} runs "
                    f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)"
                )

    df = pd.DataFrame(results)

    df = df.sort_values(
        ["initial_distribution", "adaptive_tolerance",
         "initial_preference_width", "regime", "seed"]
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    total_time = time.time() - start
    print("=" * 70)
    print(f"Done. {n_runs} runs in {total_time:.1f}s "
          f"({total_time / n_runs:.2f}s/run avg).")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
