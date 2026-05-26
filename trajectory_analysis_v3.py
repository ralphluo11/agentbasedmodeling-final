# trajectory_analysis_v3.py

"""
Long-horizon trajectory analysis under different regimes.

Records per-step trajectories under both individual-baseline and global-
trending regimes, to test whether the trending channel changes the SHAPE of
the polarization trajectory (not only its endpoint).
"""

import time
import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from model import RecommendationBackfireModel


# -----------------------------
# Configuration
# -----------------------------

WIDTHS = [0.20, 1.00]
REGIMES = [
    (0.0, 0.0),    # individual baseline
    (0.0, 0.5),    # global trending
]
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

OUTPUT_PATH_TRAJECTORIES = "trajectory_results_v3.csv"
OUTPUT_PATH_DIAGNOSTICS = "trajectory_diagnostics_v3.csv"


def regime_label(s, t):
    if s == 0.0 and t == 0.0:
        return "individual_baseline"
    if s == 0.0 and t > 0:
        return "global_trending"
    return f"social={s}_trending={t}"


def run_one(args):
    """Run one model and return its full per-step trajectory."""
    width, sw, tw, distribution, seed = args

    model = RecommendationBackfireModel(
        initial_preference_width=width,
        adaptive_tolerance=ADAPTIVE,
        initial_distribution=distribution,
        social_signal_weight=sw,
        trending_weight=tw,
        seed=seed,
        **FIXED_PARAMS,
    )
    model.run_model(N_STEPS)

    df = model.datacollector.get_model_vars_dataframe().reset_index()
    df = df.rename(columns={"index": "step"})
    df["initial_preference_width"] = width
    df["social_signal_weight"] = sw
    df["trending_weight"] = tw
    df["regime"] = regime_label(sw, tw)
    df["initial_distribution"] = distribution
    df["seed"] = seed
    return df


def compute_diagnostics(df):
    """
    For each (regime, distribution, width) cell, compute trajectory diagnostics:
        peak_step, peak_value, end_value, relaxation, late_slope_per_100_steps.
    """
    diagnostics = []

    for regime in df["regime"].unique():
        for dist in DISTRIBUTIONS:
            for w in WIDTHS:
                sub = df[
                    (df["regime"] == regime)
                    & (df["initial_distribution"] == dist)
                    & (df["initial_preference_width"] == w)
                ]
                if len(sub) == 0:
                    continue
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
                    "regime": regime,
                    "distribution": dist,
                    "width": w,
                    "peak_step": peak_step,
                    "peak_value": round(peak_value, 4),
                    "end_value": round(end_value, 4),
                    "relaxation": round(relaxation, 4),
                    "late_slope_per_100_steps": round(late_slope_per_100, 4),
                })

    return pd.DataFrame(diagnostics)


def build_arg_list():
    seeds = list(range(1, N_SEEDS + 1))
    args = []
    for width, (sw, tw), distribution, seed in itertools.product(
        WIDTHS, REGIMES, DISTRIBUTIONS, seeds
    ):
        args.append((width, sw, tw, distribution, seed))
    return args


def main():
    args_list = build_arg_list()
    n_runs = len(args_list)
    n_workers = max(1, cpu_count() - 1)

    print("=" * 70)
    print("Experiment C: Long-Horizon Trajectory v3 (with regimes)")
    print("=" * 70)
    print(f"Widths:                {WIDTHS}")
    print(f"Regimes (social,trend):{REGIMES}")
    print(f"Distributions:         {DISTRIBUTIONS}")
    print(f"Seeds per cell:        {N_SEEDS}")
    print(f"Steps per run:         {N_STEPS}")
    print(f"Total runs:            {n_runs}")
    print(f"Workers:               {n_workers}")
    print("=" * 70)

    start = time.time()
    with Pool(processes=n_workers) as pool:
        rows = []
        for i, df in enumerate(pool.imap_unordered(run_one, args_list), start=1):
            rows.append(df)
            if i % 20 == 0 or i == n_runs:
                elapsed = time.time() - start
                rate = i / elapsed
                remaining = (n_runs - i) / rate if rate > 0 else 0
                print(f"  {i}/{n_runs} ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUTPUT_PATH_TRAJECTORIES, index=False)
    print(f"\nSaved trajectories to: {OUTPUT_PATH_TRAJECTORIES} ({len(df)} rows)")

    diag = compute_diagnostics(df)
    diag.to_csv(OUTPUT_PATH_DIAGNOSTICS, index=False)
    print(f"Saved diagnostics to: {OUTPUT_PATH_DIAGNOSTICS}")
    print()
    print("Diagnostics summary:")
    print(diag.to_string(index=False))

    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
