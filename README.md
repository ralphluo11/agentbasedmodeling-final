# When Diverse Exposure Backfires

An agent-based model of political opinion dynamics in algorithmic recommendation systems. The model examines when a global "trending" channel — which pushes the same content to every user regardless of personalization — depolarizes a population, and when it instead amplifies polarization through user rejection and backfire.

Companion code for the paper *When Exposure Backfires: User Rejection, Recommendation Feedback, and Political Polarization*.

## Research question

Under what conditions does a global algorithmic trending channel amplify political polarization, and when does it reduce it?

## Files

### Model
- `agents.py` — `UserAgent` class: opinion updating, accept/ignore/reject behavior, and adaptive tolerance updates.
- `model.py` — `RecommendationBackfireModel` class: initialization, three recommendation channels (individual feedback learning, local collaborative filtering, global trending), and data collection.

### Interactive app
- `app.py` — Solara GUI that streams the simulation step-by-step. Plots update in real time as the model runs.

### Batch experiments
- `batch_run_1.py` — main batch: sweeps recommendation width, adaptive tolerance, initial distribution, and four cross-user signaling regimes (1,440 runs).
- `backfire_causal.py` — backfire-rate ablation: tests whether polarization depends on the rejection-backfire channel (960 runs).
- `trajectory_analysis_v3.py` — long-horizon trajectories: 400-step runs comparing the individual baseline to pure global trending (240 runs).
- `cf_density_sweep.py` — fine-grained sweep over the local collaborative-filtering weight at trending = 0 (600 runs).

### Analysis
- `section6_analysis.ipynb` — Jupyter notebook that reads the four CSV outputs and generates the four figures and two tables used in the paper.

## Installation

```bash
pip install mesa solara pandas numpy matplotlib jupyter
```

## Run the interactive app

```bash
solara run app.py
```

The app exposes every model parameter as a slider, plus an animation panel (step delay, plot-refresh interval) for controlling how quickly the simulation plays. Both cross-user channels are active by default (local CF and global trending at 0.3 each) so first-time users immediately see the cross-user dynamics. Set either weight to 0 to disable that channel.

## Run the batch experiments

Each experiment uses multiprocessing and reports progress to the terminal.

### Main batch
```bash
python batch_run_1.py
```
Output: `batch_results_v2.csv`

### Backfire-rate ablation
```bash
python backfire_causal.py
```
Output: `backfire_causal_results_v2.csv`

### Long-horizon trajectory
```bash
python trajectory_analysis_v3.py
```
Output: `trajectory_results_v3.csv`

### Local CF density sweep
```bash
python cf_density_sweep.py
```
Output: `cf_density_results.csv`

## Generate figures and tables

After the four CSV files exist, open the notebook and run all cells:

```bash
jupyter notebook section6_analysis.ipynb
```

Outputs:
- `fig1_regime_width_distribution.png` — final mean extremity by regime, width, and initial distribution (the headline width-conditional reversal)
- `fig2_cf_density.png` — local CF sweep showing no detectable effect
- `fig3_backfire_ablation.png` — backfire sweep showing trending-channel amplification depends on rejection movement
- `fig4_trajectory_shapes.png` — 400-step trajectories under baseline vs. trending
- `table_a_adaptive_buffer.csv` — adaptive-tolerance buffer effect by distribution × width × regime
- `table_b_trajectory_diagnostics.csv` — peak step, peak value, end value, relaxation, and late-stage slope for each cell

## Notes

- The model uses 200 agents. The main batch and ablation experiments run for 100 steps per cell; the trajectory experiment runs for 400 steps. Every cell of every batch is replicated with 20 random seeds.
- The headline outcome metric is **mean extremity** — the average absolute distance of agents' opinions from the ideological center. Other reporters collected at every step include signed mean opinion, opinion variance, extreme share, recommendation widths, thresholds, accept/ignore/reject rates, and trending-pool diagnostics.
- Numerical constants in the model are theoretically motivated rather than empirically calibrated, following pattern-oriented modeling (Grimm et al. 2005). Section 4 of the paper documents the three independent empirical anchors that justify the parameter choices.