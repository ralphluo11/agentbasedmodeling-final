# app.py

"""
This app allows you to change model parameters, run the simulation,
and visualize model outcomes while the simulation progresses.

Core mechanism:

    content recommendation
        -> psychological response
        -> user feedback
        -> algorithmic update
        -> future exposure
        -> aggregate polarization
"""

import time

import solara
import matplotlib.pyplot as plt

from model import RecommendationBackfireModel


# -----------------------------
# Plot helpers
# -----------------------------

# Semantic color palette: each plot family uses a distinct color so the reader
# can tell at a glance which dimension of the simulation a plot is about.
COLOR_OPINION    = "#6f42c1"  # purple   — user opinions / extremity
COLOR_ALGORITHM  = "#ff7f0e"  # orange   — algorithm-side (preference width, recommender)
COLOR_CROSS_USER = "#17a2b8"  # teal     — cross-user signals (trending pool, exposure, history)
COLOR_ACCEPT     = "#2ca02c"  # green    — accept response
COLOR_IGNORE     = "#7f7f7f"  # gray     — ignore response
COLOR_REJECT     = "#d62728"  # red      — reject / threshold boundaries


def make_line_plot(model_data, y_column, title, y_label, hline=None,
                   color=COLOR_OPINION):
    """
    Single-line plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(model_data.index, model_data[y_column], color=color, linewidth=1.8)
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"initial = {hline:.2f}")
        ax.legend(loc="best", fontsize=9, frameon=False)
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_feedback_rates_plot(model_data):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model_data.index, model_data["Acceptance Rate"],
            label="Accept", color=COLOR_ACCEPT, linewidth=1.8)
    ax.plot(model_data.index, model_data["Ignore Rate"],
            label="Ignore", color=COLOR_IGNORE, linewidth=1.8)
    ax.plot(model_data.index, model_data["Backfire Rate"],
            label="Reject (backfire)", color=COLOR_REJECT, linewidth=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Share of all exposures so far")
    ax.set_title("How users are responding to recommendations")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_opinion_vs_algorithm_plot(model_data):
    """
    Two-line plot showing whether the algorithm's learned preference center
    is co-evolving with user opinion (lines move together) or stays anchored
    to its initial value (only the user line moves).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model_data.index, model_data["Mean Extremity"],
            label="User opinion extremity",
            color=COLOR_OPINION, linewidth=1.8)
    ax.plot(model_data.index, model_data["Mean Preference Center Extremity"],
            label="Algorithm's learned preference extremity",
            color=COLOR_ALGORITHM, linewidth=1.8, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean distance from center")
    ax.set_title("Is the algorithm tracking the users?")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_exposure_distance_histogram(model):
    """
    Histogram of each agent's last ideological distance from the content
    they were just shown. This is the visual signature of the backfire
    mechanism: when the trending channel is active and recommendations
    are wide, the right tail of this distribution thickens — many users
    are seeing content beyond their rejection threshold, which is what
    aggregates into population-level polarization.
    """
    distances = [a.last_distance for a in model.agents
                 if a.last_distance is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(distances) > 0:
        ax.hist(distances, bins=25, color=COLOR_CROSS_USER, alpha=0.85)
        # Mark the rejection-zone boundaries (low and high tolerance).
        ax.axvline(model.low_rejection_threshold, color=COLOR_REJECT,
                   linestyle="--", linewidth=1.2,
                   label=f"low-tolerance reject threshold = "
                         f"{model.low_rejection_threshold:.2f}")
        ax.axvline(model.high_rejection_threshold, color=COLOR_REJECT,
                   linestyle=":", linewidth=1.2,
                   label=f"high-tolerance reject threshold = "
                         f"{model.high_rejection_threshold:.2f}")
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_xlabel("Ideological distance between agent and content")
    ax.set_ylabel("Number of agents")
    ax.set_title("Are users seeing content inside or beyond their tolerance?")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_opinion_histogram(model):
    """
    Distribution of agent opinions on [-1, +1]. Bimodal peaks indicate
    polarization; concentration near zero indicates moderation.
    """
    opinions = [agent.opinion for agent in model.agents]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(opinions, bins=25, range=(-1.0, 1.0),
            color=COLOR_OPINION, alpha=0.85)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Opinion (-1 to +1)")
    ax.set_ylabel("Number of agents")
    ax.set_title("Where is the population on the ideological spectrum?")
    ax.set_xlim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_preference_width_histogram(model):
    """
    Distribution of per-agent recommendation widths. Compare against the
    initial width (vertical dashed line): a leftward shift means the
    algorithm has narrowed exposure; a rightward shift means it has
    broadened.
    """
    widths = [agent.preference_width for agent in model.agents]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(widths, bins=25, range=(0.0, 1.0),
            color=COLOR_ALGORITHM, alpha=0.85)
    ax.axvline(model.initial_preference_width, color="gray", linestyle="--",
               linewidth=1.2, alpha=0.7,
               label=f"initial = {model.initial_preference_width:.2f}")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.set_xlabel("Recommendation width")
    ax.set_ylabel("Number of agents")
    ax.set_title("How narrow has the algorithm made each user's exposure?")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_summary_items(model_data):
    final = model_data.iloc[-1]
    return [
        ("Mean Opinion", final["Mean Opinion"], "Signed average; near 0 means the population is balanced between left and right."),
        ("Mean Extremity", final["Mean Extremity"], "Average |opinion|; how far from the center, regardless of side. High = polarized."),
        ("Opinion Variance", final["Opinion Variance"], "Spread of opinions across the population."),
        ("Extreme Share", final["Extreme Share"], "Fraction of agents with |opinion| ≥ 0.75."),
        ("Mean Preference Center", final["Mean Preference Center"], "Signed average of the algorithm's learned preference centers."),
        ("Preference Center Extremity", final["Mean Preference Center Extremity"], "How extreme the algorithm's learned profiles have become."),
        ("Mean Preference Width", final["Mean Preference Width"], "Average breadth of recommendation exposure. Lower = more personalized."),
        ("Mean Acceptance Threshold", final["Mean Acceptance Threshold"], "Average distance within which users accept content."),
        ("Mean Rejection Threshold", final["Mean Rejection Threshold"], "Average distance beyond which users reject content."),
        ("Acceptance Rate", final["Acceptance Rate"], "Cumulative share of exposures users accepted."),
        ("Ignore Rate", final["Ignore Rate"], "Cumulative share of exposures users ignored."),
        ("Backfire Rate", final["Backfire Rate"], "Cumulative share of exposures users rejected."),
        ("Average Exposure Distance", final["Average Exposure Distance"], "Average ideological distance between user and recommended content."),
    ]


@solara.component
def SummaryGrid(model_data):
    items = make_summary_items(model_data)
    left_items = items[:7]
    right_items = items[7:]

    def compact_metric_row(name, value, description):
        with solara.Row(
            gap="8px",
            style={
                "padding": "4px 0",
                "border-bottom": "1px solid #eeeeee",
                "align-items": "baseline",
            },
        ):
            solara.Markdown(
                f"""<div style="width: 260px; font-weight: 600; font-size: 14px;">{name}</div>""",
                unsafe_solara_execute=True,
            )
            solara.Markdown(
                f"""<div style="width: 70px; font-family: monospace; font-size: 14px;">{value:.4f}</div>""",
                unsafe_solara_execute=True,
            )
            solara.Markdown(
                f"""<div style="color: #666; font-size: 13px;">{description}</div>""",
                unsafe_solara_execute=True,
            )

    with solara.Columns([1, 1], gutters=True):
        with solara.Column(gap="0px"):
            for name, value, description in left_items:
                compact_metric_row(name, value, description)
        with solara.Column(gap="0px"):
            for name, value, description in right_items:
                compact_metric_row(name, value, description)


# -----------------------------
# Reactive state
# -----------------------------

num_agents_state = solara.reactive(200)
steps_state = solara.reactive(150)
initial_preference_width_state = solara.reactive(0.45)
high_tolerance_share_state = solara.reactive(0.50)
adaptive_tolerance_state = solara.reactive(True)
assimilation_rate_state = solara.reactive(0.08)
backfire_rate_state = solara.reactive(0.06)
initial_distribution_state = solara.reactive("polarized")
seed_state = solara.reactive(42)

social_signal_weight_state = solara.reactive(0.3)
trending_weight_state = solara.reactive(0.3)

# Streaming-related state
step_delay_state = solara.reactive(0.05)  # seconds between steps (animation speed)
update_every_state = solara.reactive(1)   # collect+rerender every N steps

# Live simulation state
model_state = solara.reactive(None)
model_data_state = solara.reactive(None)
current_step_state = solara.reactive(0)
is_running_state = solara.reactive(False)
should_stop_state = solara.reactive(False)
has_started_state = solara.reactive(False)


# -----------------------------
# Streaming simulation logic
# -----------------------------

def stream_simulation():
    """
    Run the simulation step-by-step, updating reactive state as it progresses.
    This allows the UI to update in real time as the model runs.
    """

    sw = social_signal_weight_state.value
    tw = trending_weight_state.value
    if sw + tw > 1.0:
        tw = max(0.0, 1.0 - sw)
        trending_weight_state.value = tw

    model = RecommendationBackfireModel(
        num_agents=num_agents_state.value,
        initial_distribution=initial_distribution_state.value,
        high_tolerance_share=high_tolerance_share_state.value,
        feedback_sensitivity=0.25,
        initial_preference_width=initial_preference_width_state.value,
        adaptive_tolerance=adaptive_tolerance_state.value,
        assimilation_rate=assimilation_rate_state.value,
        backfire_rate=backfire_rate_state.value,
        social_signal_weight=sw,
        trending_weight=tw,
        seed=seed_state.value,
    )

    model_state.value = model
    model_data_state.value = model.datacollector.get_model_vars_dataframe()
    current_step_state.value = 0
    is_running_state.value = True
    should_stop_state.value = False
    has_started_state.value = True

    n_steps = steps_state.value
    update_every = max(1, update_every_state.value)
    delay = max(0.0, step_delay_state.value)

    for step in range(1, n_steps + 1):
        if should_stop_state.value:
            break
        model.step()
        current_step_state.value = step

        # Refresh the dataframe at the configured cadence so plots update
        # smoothly without re-rendering on every single steps.
        if step % update_every == 0 or step == n_steps:
            model_data_state.value = model.datacollector.get_model_vars_dataframe()

        if delay > 0:
            time.sleep(delay)

    # Final refresh to make sure the last step is shown
    model_data_state.value = model.datacollector.get_model_vars_dataframe()
    is_running_state.value = False


# -----------------------------
# Solara Page
# -----------------------------

@solara.component
def Page():
    solara.Title("When Diverse Exposure Backfires")

    # Background task: only runs when triggered. Solara handles lifecycle.
    sim_task = solara.lab.use_task(
        stream_simulation,
        dependencies=None,  
    )

    with solara.Column(gap="16px"):
        solara.Markdown(
            """
# When Diverse Exposure Backfires

**Research question:**
Under what conditions does a global algorithmic trending channel amplify political polarization, and when does it reduce it?

The plots below update in real time as the simulation runs. You can watch how
user opinions, the algorithm's learned preferences, and feedback rates co-evolve
step by step rather than seeing only the final state.
"""
        )

        with solara.Sidebar():
            solara.Markdown("## Model Parameters")
            solara.Markdown(
                """
Use these controls to change the simulated recommendation environment.
After clicking **Run**, the simulation will animate step by step.
"""
            )

            solara.Markdown("---")

            solara.Markdown("### 1. Population")
            solara.SliderInt("Number of agents", value=num_agents_state,
                             min=50, max=500, step=50)
            solara.SliderInt("Simulation steps", value=steps_state,
                             min=25, max=400, step=25)
            solara.Select(
                label="Initial opinion distribution",
                value=initial_distribution_state,
                values=["polarized", "uniform", "moderate"],
            )

            solara.Markdown("---")

            solara.Markdown("### 2. Recommendation Algorithm")
            solara.SliderFloat("Initial recommendation width",
                               value=initial_preference_width_state,
                               min=0.05, max=1.00, step=0.05)
            solara.SliderFloat("Local CF weight (social signal)",
                               value=social_signal_weight_state,
                               min=0.0, max=0.9, step=0.05)
            solara.SliderFloat("Global trending weight",
                               value=trending_weight_state,
                               min=0.0, max=0.9, step=0.05)
            solara.Markdown(
                "_Local CF + global trending must sum to at most 1.0._"
            )

            solara.Markdown("---")

            solara.Markdown("### 3. User Psychology")
            solara.SliderFloat("High-tolerance agent share",
                               value=high_tolerance_share_state,
                               min=0.00, max=1.00, step=0.05)
            solara.Checkbox(label="Adaptive tolerance",
                            value=adaptive_tolerance_state)
            solara.SliderFloat("Assimilation rate",
                               value=assimilation_rate_state,
                               min=0.01, max=0.30, step=0.01)
            solara.SliderFloat("Backfire rate",
                               value=backfire_rate_state,
                               min=0.01, max=0.30, step=0.01)

            solara.Markdown("---")

            solara.Markdown("### 4. Reproducibility")
            solara.SliderInt("Random seed", value=seed_state,
                             min=1, max=999, step=1)

            solara.Markdown("---")

            solara.Markdown("### 5. Animation")
            solara.SliderFloat("Step delay (seconds)",
                               value=step_delay_state,
                               min=0.00, max=0.30, step=0.01)
            solara.Markdown(
                "_Higher delay = slower animation, easier to watch the dynamics develop._"
            )
            solara.SliderInt("Refresh plots every N steps",
                             value=update_every_state,
                             min=1, max=10, step=1)
            solara.Markdown(
                "_Lower refresh = smoother animation but slower; higher = chunky but faster._"
            )

            solara.Markdown("---")

            # Run / Stop buttons
            def on_run():
                # Reset state for a fresh run
                should_stop_state.value = False
                sim_task()

            def on_stop():
                should_stop_state.value = True

            with solara.Row():
                solara.Button(
                    "Run simulation",
                    on_click=on_run,
                    color="primary",
                    disabled=is_running_state.value,
                )
                solara.Button(
                    "Stop",
                    on_click=on_stop,
                    color="secondary",
                    disabled=not is_running_state.value,
                )

        # -----------------------------
        # Main panel
        # -----------------------------

        if not has_started_state.value:
            solara.Info(
                "Set the parameters in the sidebar, then click **Run simulation** "
                "to watch the model develop step by step."
            )
            return

        model = model_state.value
        model_data = model_data_state.value

        if model is None or model_data is None or len(model_data) == 0:
            solara.Info("Initializing simulation...")
            return

        # Progress display
        n_steps = steps_state.value
        progress = current_step_state.value / max(1, n_steps)
        status = "Running" if is_running_state.value else "Completed"
        if should_stop_state.value and not is_running_state.value:
            status = "Stopped"

        solara.Markdown(
            f"### {status}: step {current_step_state.value} / {n_steps}"
        )
        solara.ProgressLinear(value=progress * 100, color="primary")

        solara.Markdown("## Current Summary")
        SummaryGrid(model_data)

        solara.Markdown("## Process: how users and the algorithm are interacting")

        with solara.Columns([1, 1]):
            with solara.Column():
                fig_process_1 = make_opinion_vs_algorithm_plot(model_data)
                solara.FigureMatplotlib(fig_process_1)
                solara.Markdown(
                    "**Read:** Both lines rising together means the algorithm is chasing "
                    "the users, not pulling them anywhere. Divergence would mean the "
                    "algorithm has its own drift."
                )

            with solara.Column():
                fig_process_2 = make_feedback_rates_plot(model_data)
                solara.FigureMatplotlib(fig_process_2)
                solara.Markdown(
                    "**Read:** Cumulative share of each response across all exposures so "
                    "far. A rising reject share is the signal that the recommender is "
                    "serving content beyond users' tolerance."
                )

        solara.Markdown("## Outcomes: what is happening to the population")

        initial_width = initial_preference_width_state.value

        with solara.Columns([1, 1]):
            with solara.Column():
                fig1 = make_line_plot(
                    model_data, "Mean Extremity",
                    "Average ideological extremity",
                    "Mean |opinion|",
                    color=COLOR_OPINION,
                )
                solara.FigureMatplotlib(fig1)
                solara.Markdown(
                    "**Read:** The mean distance from the ideological center, "
                    "averaged over all agents. Rising = polarizing; falling = depolarizing."
                )

            with solara.Column():
                fig4 = make_line_plot(
                    model_data, "Mean Preference Width",
                    "How broad is each user's content diet?",
                    "Mean recommendation width",
                    hline=initial_width,
                    color=COLOR_ALGORITHM,
                )
                solara.FigureMatplotlib(fig4)
                solara.Markdown(
                    "**Read:** Average breadth of the recommendation distribution. "
                    "The dashed line marks the initial width. Falling below it means the "
                    "algorithm is narrowing each user's exposure."
                )

        solara.Markdown("## Cross-user signals: what the trending channel is doing")
        solara.Markdown(
            "These plots are most informative when **Local CF weight** or "
            "**Global trending weight** is above zero. They expose how cross-user "
            "feedback shapes the content every user sees."
        )

        with solara.Columns([1, 1]):
            with solara.Column():
                fig_exposure = make_exposure_distance_histogram(model)
                solara.FigureMatplotlib(fig_exposure)
                solara.Markdown(
                    "**Read:** Distribution of the ideological distance between each "
                    "agent and the content they were just shown. Mass beyond a rejection "
                    "threshold (dashed/dotted red) is content that will trigger backfire — "
                    "the visual signature of how the trending channel amplifies "
                    "polarization at wide recommendation widths."
                )

            with solara.Column():
                fig_hist = make_line_plot(
                    model_data, "Mean Acceptance History Filled",
                    "How much behavioral data does the algorithm have?",
                    "Mean fill (0–1)",
                    color=COLOR_CROSS_USER,
                )
                solara.FigureMatplotlib(fig_hist)
                solara.Markdown(
                    "**Read:** Average fill of each user's recent-acceptance memory. "
                    "Cross-user signals are weak early in a run (memories empty) and "
                    "stronger once users have built up acceptance histories."
                )

        with solara.Columns([1, 1]):
            with solara.Column():
                fig_trend_mean = make_line_plot(
                    model_data, "Trending Pool Mean",
                    "What ideology is the trending pool pushing?",
                    "Mean ideology of trending pool",
                    color=COLOR_CROSS_USER,
                )
                solara.FigureMatplotlib(fig_trend_mean)
                solara.Markdown(
                    "**Read:** Mean ideology of the global pool that the trending "
                    "channel injects into every user. Drift away from zero indicates "
                    "one side is dominating the population's accepted content. "
                    "Flat at zero when trending weight is zero."
                )

            with solara.Column():
                fig_trend_std = make_line_plot(
                    model_data, "Trending Pool Std",
                    "How diverse is the trending pool?",
                    "Std. dev. of trending pool",
                    color=COLOR_CROSS_USER,
                )
                solara.FigureMatplotlib(fig_trend_std)
                solara.Markdown(
                    "**Read:** Ideological spread of the trending pool. Smaller values "
                    "mean the pool has consolidated around a narrow band of content."
                )

        solara.Markdown("## Snapshot: where the population stands now")

        with solara.Columns([1, 1]):
            with solara.Column():
                fig5 = make_opinion_histogram(model)
                solara.FigureMatplotlib(fig5)
                solara.Markdown(
                    "**Read:** Current distribution of agent opinions. Bimodal peaks "
                    "near the ends are the diagnostic of polarization; a single peak "
                    "near zero is depolarization."
                )

            with solara.Column():
                fig6 = make_preference_width_histogram(model)
                solara.FigureMatplotlib(fig6)
                solara.Markdown(
                    "**Read:** Distribution of per-agent recommendation widths. "
                    "Compare to the dashed initial-width line: mass to the left means "
                    "the algorithm has narrowed exposure for most users."
                )

        if not is_running_state.value:
            solara.Markdown("## Raw Model Data")
            solara.DataFrame(model_data.reset_index(), items_per_page=10)
