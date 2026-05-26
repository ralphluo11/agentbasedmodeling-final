# cf_density_sweep.py

"""
Fine-grained sweep over social_signal_weight.

In the main batch we only tested social_signal_weight in {0.0, 0.5}. This
experiment fills in the gap with five values from 0 to 0.8, holding
trending_weight at 0, to test whether local collaborative filtering has any
detectable effect at higher weights and to characterize the marginal effect
of the CF channel cleanly.
"""

import time
import itertools
from multiprocessing import Pool, cpu_count

import pandas as pd

from model import RecommendationBackfireModel


# -----------------------------
# Configuration
# -----------------------------

WIDTHS = [0.20, 1.00]
SOCIAL_WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8]
DISTRIBUTIONS = ["polarized", "uniform", "moderate"]
ADAPTIVE = False

N_SEEDS = 20
N_STEPS = 100

FIXED_PARAMS = {
    "num_agents": 200,
    "high_tolerance_share": 0.5,
    "feedback_sensitivity": 0.25,
    "assimilation_rate": 0.08,
    "backfire_rate": 0.06,
    "trending_weight": 0.0,
}

OUTPUT_PATH = "cf_density_results.csv"


def run_one(args):
    width, sw, distribution, seed = args

    model = RecommendationBackfireModel(
        initial_preference_width=width,
        adaptive_tolerance=ADAPTIVE,
        initial_distribution=distribution,
        social_signal_weight=sw,
        seed=seed,
        **FIXED_PARAMS,
    )
    model.run_model(N_STEPS)

    df = model.datacollector.get_model_vars_dataframe()
    final = df.iloc[-1]
    initial = df.iloc[0]

    return {
        "initial_preference_width": width,
        "social_signal_weight": sw,
        "initial_distribution": distribution,
        "seed": seed,
        "initial_mean_extremity": initial["Mean Extremity"],
        "final_mean_extremity": final["Mean Extremity"],
        "final_opinion_variance": final["Opinion Variance"],
        "final_extreme_share": final["Extreme Share"],
        "final_preference_width": final["Mean Preference Width"],
        "final_acceptance_rate": final["Acceptance Rate"],
        "final_backfire_rate": final["Backfire Rate"],
        "final_opinion_bimodality": final["Opinion Bimodality"],
    }


def build_arg_list():
    seeds = list(range(1, N_SEEDS + 1))
    args = []
    for width, sw, distribution, seed in itertools.product(
        WIDTHS, SOCIAL_WEIGHTS, DISTRIBUTIONS, seeds
    ):
        args.append((width, sw, distribution, seed))
    return args


def main():
    args_list = build_arg_list()
    n_runs = len(args_list)
    n_workers = max(1, cpu_count() - 1)

    print("=" * 70)
    print("Experiment D: CF Density Sweep")
    print("=" * 70)
    print(f"Widths:                {WIDTHS}")
    print(f"social_signal_weights: {SOCIAL_WEIGHTS}")
    print(f"Distributions:         {DISTRIBUTIONS}")
    print(f"trending_weight:       0.0 (fixed)")
    print(f"Adaptive tolerance:    {ADAPTIVE} (fixed)")
    print(f"Seeds per cell:        {N_SEEDS}")
    print(f"Total runs:            {n_runs}")
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
                print(f"  {i}/{n_runs} ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")

    df = pd.DataFrame(results)
    df = df.sort_values(
        ["initial_distribution", "initial_preference_width", "social_signal_weight", "seed"]
    ).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    total_time = time.time() - start
    print("=" * 70)
    print(f"Done. {n_runs} runs in {total_time:.1f}s")
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
