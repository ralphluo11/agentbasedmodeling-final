# backfire_causal_v2.py

"""
Experiment B: Backfire-causal experiment v2 (with collaborative + trending channels).

Tests whether the rejection-backfire channel is necessary for polarization
under different recommendation regimes (individual baseline / local CF /
global trending / hybrid).

Sweep:
    backfire_rate         : {0.00, 0.03, 0.06, 0.12}
    initial_preference_width : {0.20, 0.50, 1.00}
    regime (social, trending):
        (0.0, 0.0)  individual_baseline
        (0.5, 0.0)  local_cf
        (0.0, 0.5)  global_trending
        (0.4, 0.4)  hybrid
    seed                  : 1..20

Initial distribution fixed at uniform (cleanest test of emergent polarization).
Adaptive tolerance fixed at False (so we isolate the backfire channel cleanly).

Total: 4 * 3 * 4 * 20 = 960 runs * 100 steps.

Run with:
    python backfire_causal_v2.py
"""

import time
import itertools
from multiprocessing import Pool, cpu_count

import pandas as pd

from model import RecommendationBackfireModel


# -----------------------------
# Configuration
# -----------------------------

BACKFIRE_RATES = [0.00, 0.03, 0.06, 0.12]
WIDTHS = [0.20, 0.50, 1.00]

REGIMES = [
    (0.0, 0.0),    # individual baseline
    (0.5, 0.0),    # local CF
    (0.0, 0.5),    # global trending
    (0.4, 0.4),    # hybrid
]

INITIAL_DISTRIBUTION = "uniform"
ADAPTIVE = False
N_SEEDS = 20
N_STEPS = 100

FIXED_PARAMS = {
    "num_agents": 200,
    "high_tolerance_share": 0.5,
    "feedback_sensitivity": 0.25,
    "assimilation_rate": 0.08,
}

OUTPUT_PATH = "backfire_causal_results_v2.csv"


def regime_label(s, t):
    if s == 0.0 and t == 0.0:
        return "individual_baseline"
    if s > 0 and t == 0.0:
        return "local_cf"
    if s == 0.0 and t > 0:
        return "global_trending"
    return "hybrid"


def run_one(args):
    backfire_rate, width, sw, tw, seed = args

    model = RecommendationBackfireModel(
        initial_preference_width=width,
        adaptive_tolerance=ADAPTIVE,
        initial_distribution=INITIAL_DISTRIBUTION,
        backfire_rate=backfire_rate,
        social_signal_weight=sw,
        trending_weight=tw,
        seed=seed,
        **FIXED_PARAMS,
    )
    model.run_model(N_STEPS)

    df = model.datacollector.get_model_vars_dataframe()
    final = df.iloc[-1]
    initial = df.iloc[0]

    return {
        "backfire_rate": backfire_rate,
        "initial_preference_width": width,
        "social_signal_weight": sw,
        "trending_weight": tw,
        "regime": regime_label(sw, tw),
        "seed": seed,
        "initial_mean_extremity": initial["Mean Extremity"],
        "final_mean_extremity": final["Mean Extremity"],
        "final_opinion_variance": final["Opinion Variance"],
        "final_extreme_share": final["Extreme Share"],
        "final_acceptance_rate": final["Acceptance Rate"],
        "final_backfire_rate": final["Backfire Rate"],
        "final_ignore_rate": final["Ignore Rate"],
        "final_opinion_bimodality": final["Opinion Bimodality"],
    }


def build_arg_list():
    seeds = list(range(1, N_SEEDS + 1))
    args = []
    for br, w, (sw, tw), seed in itertools.product(
        BACKFIRE_RATES, WIDTHS, REGIMES, seeds
    ):
        args.append((br, w, sw, tw, seed))
    return args


def main():
    args_list = build_arg_list()
    n_runs = len(args_list)
    n_workers = max(1, cpu_count() - 1)

    print("=" * 70)
    print("Experiment B: Backfire-Causal v2 (with CF + trending regimes)")
    print("=" * 70)
    print(f"Backfire rates:        {BACKFIRE_RATES}")
    print(f"Widths:                {WIDTHS}")
    print(f"Regimes (social,trend):{REGIMES}")
    print(f"Initial distribution:  {INITIAL_DISTRIBUTION} (fixed)")
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
        ["regime", "initial_preference_width", "backfire_rate", "seed"]
    ).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    total_time = time.time() - start
    print("=" * 70)
    print(f"Done. {n_runs} runs in {total_time:.1f}s")
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
