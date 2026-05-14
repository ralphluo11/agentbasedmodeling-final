# Recommendation Backfire ABM

This project is an agent-based model of political opinion dynamics in algorithmic recommendation systems. It examines when broader ideological content recommendations reduce polarization and when they instead create rejection-driven backfire.

## Research question

Under what conditions can algorithmic content diversity produce political polarization independent of social-network effects and user-driven content selection, and does adaptive psychological tolerance buffer or amplify this effect?

## Files

- `agents.py` — defines the user agents, opinion updating rules, acceptance/ignore/rejection behavior, and adaptive tolerance updates.
- `model.py` — defines the Mesa model, initialization rules, recommendation process, and data collection.
- `app.py` — launches the interactive Solara GUI for exploring model behavior.
- `batch_run_1.py` — runs the main batch experiment over recommendation width, adaptive tolerance, and initial opinion distribution.
- `batch_analysis_1.py` — analyzes `batch_results.csv` and generates summary figures.
- `backfire_causal.py` — runs the backfire-rate ablation experiment.
- `trajectory_analysis_v2.py` — runs the long-horizon trajectory experiment.
- `ABM Final Draft.pdf` — final project paper.

## Installation

Install the required Python packages:

```bash
pip install mesa solara pandas numpy matplotlib
```

## Run the GUI

```bash
solara run app.py
```

The GUI lets users vary key model parameters, including recommendation width, adaptive tolerance, assimilation rate, and backfire rate. Default values match the main parameter choices used in the paper.

## Run the batch experiments

Main batch experiment:

```bash
python batch_run_1.py
```

This creates:

```text
batch_results.csv
```

Analyze the main batch results:

```bash
python batch_analysis_1.py
```

This generates figures such as:

```text
fig1_main_extremity_by_dist.png
fig3_delta_extremity.png
fig5_threshold_drift_by_dist.png
fig7_essay_main.png
```

Backfire causal experiment:

```bash
python backfire_causal.py
```

This creates:

```text
backfire_causal_results.csv
fig_backfire_causal.png
```

Long-horizon trajectory experiment:

```bash
python trajectory_analysis_v2.py
```

This creates:

```text
trajectory_results_v2.csv
trajectory_diagnostics.csv
fig_trajectories_v2.png
fig_overshoot_focus.png
```

## Notes

The main model uses 200 agents. In the reported experiments, the main batch runs for 100 steps, while the trajectory experiment runs for 400 steps. The main outcome is mean ideological extremity, measured as the average absolute distance of agents' opinions from the ideological center.

