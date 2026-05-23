# app.py

"""
Solara app for the recommendation-backfire ABM.

Run with:

    solara run app.py

This app allows you to change model parameters, run the simulation,
and visualize model outcomes.

Core mechanism:

    content recommendation
        -> psychological response
        -> user feedback
        -> algorithmic update
        -> future exposure
        -> aggregate polarization
"""

import solara
import pandas as pd
import matplotlib.pyplot as plt

from model import RecommendationBackfireModel


# -----------------------------
# Helper functions
# -----------------------------

def run_simulation(
    num_agents,
    steps,
    initial_preference_width,
    high_tolerance_share,
    adaptive_tolerance,
    assimilation_rate,
    backfire_rate,
    initial_distribution,
    social_signal_weight,
    trending_weight,
    seed,
):
    """
    Create and run one model instance.
    """

    model = RecommendationBackfireModel(
        num_agents=num_agents,
        initial_distribution=initial_distribution,
        high_tolerance_share=high_tolerance_share,
        feedback_sensitivity=0.25,
        initial_preference_width=initial_preference_width,
        adaptive_tolerance=adaptive_tolerance,
        assimilation_rate=assimilation_rate,
        backfire_rate=backfire_rate,
        social_signal_weight=social_signal_weight,
        trending_weight=trending_weight,
        seed=seed,
    )

    model.run_model(steps)

    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    return model, model_data, agent_data


def make_line_plot(model_data, y_column, title, y_label):
    """
    Create a simple Matplotlib line plot for one model-level variable.
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(model_data.index, model_data[y_column])
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def make_feedback_rates_plot(model_data):
    """
    Plot acceptance, ignore, and backfire rates together.

    This directly shows the behavioral feedback process:
    users accept some content, ignore some content, and reject some content.
    """

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(model_data.index, model_data["Acceptance Rate"], label="Acceptance")
    ax.plot(model_data.index, model_data["Ignore Rate"], label="Ignore")
    ax.plot(model_data.index, model_data["Backfire Rate"], label="Backfire")

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative share of exposures")
    ax.set_title("User Response Rates Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def make_opinion_vs_algorithm_plot(model_data):
    """
    Plot user opinion extremity and algorithmic preference-center extremity.

    This helps show whether the recommender's learned profile becomes more
    extreme together with the users.
    """

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        model_data.index,
        model_data["Mean Extremity"],
        label="User opinion extremity",
    )
    ax.plot(
        model_data.index,
        model_data["Mean Preference Center Extremity"],
        label="Algorithm preference extremity",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Mean absolute value")
    ax.set_title("User Opinion vs Algorithmic Preference Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def make_final_opinion_histogram(model):
    """
    Create a histogram of final agent opinions.
    """

    opinions = [agent.opinion for agent in model.agents]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(opinions, bins=25)
    ax.set_xlabel("Opinion")
    ax.set_ylabel("Number of agents")
    ax.set_title("Final Opinion Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def make_preference_width_histogram(model):
    """
    Create a histogram of final recommendation widths.
    """

    widths = [agent.preference_width for agent in model.agents]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(widths, bins=25)
    ax.set_xlabel("Recommendation width")
    ax.set_ylabel("Number of agents")
    ax.set_title("Final Recommendation Width Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def make_summary_items(model_data):
    """
    Return final summary metrics as a list of display tuples.
    This avoids Solara DataFrame pagination.
    """

    final = model_data.iloc[-1]

    return [
        ("Mean Opinion", final["Mean Opinion"], "Average signed ideology of all agents."),
        ("Mean Extremity", final["Mean Extremity"], "Average distance from the ideological center."),
        ("Opinion Variance", final["Opinion Variance"], "Dispersion of opinions across the population."),
        ("Extreme Share", final["Extreme Share"], "Share of agents near ideological poles."),
        ("Mean Preference Center", final["Mean Preference Center"], "Average algorithmic estimate of user preference."),
        ("Preference Center Extremity", final["Mean Preference Center Extremity"], "How extreme the algorithm's learned profiles are."),
        ("Mean Preference Width", final["Mean Preference Width"], "Average breadth of recommendation exposure."),
        ("Mean Acceptance Threshold", final["Mean Acceptance Threshold"], "Average boundary for accepting content."),
        ("Mean Rejection Threshold", final["Mean Rejection Threshold"], "Average boundary for rejecting content."),
        ("Acceptance Rate", final["Acceptance Rate"], "Share of recommendations accepted by users."),
        ("Ignore Rate", final["Ignore Rate"], "Share of recommendations ignored or skipped."),
        ("Backfire Rate", final["Backfire Rate"], "Share of recommendations rejected."),
        ("Average Exposure Distance", final["Average Exposure Distance"], "Average ideological distance between user and content."),
    ]


@solara.component
def SummaryGrid(model_data):
    """
    Display final summary metrics in a compact two-column layout.
    """

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
                f"""
<div style="width: 260px; font-weight: 600; font-size: 14px;">
{name}
</div>
""",
                unsafe_solara_execute=True,
            )
            solara.Markdown(
                f"""
<div style="width: 70px; font-family: monospace; font-size: 14px;">
{value:.4f}
</div>
""",
                unsafe_solara_execute=True,
            )
            solara.Markdown(
                f"""
<div style="color: #666; font-size: 13px;">
{description}
</div>
""",
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

# v2: cross-user signal channels
# Defaults are non-zero so a first-time user sees all three channels active
# (individual + local collaborative filtering + global trending) and can
# observe the cross-user dynamics in the plots. Set to 0 to disable.
social_signal_weight_state = solara.reactive(0.3)
trending_weight_state = solara.reactive(0.3)

model_state = solara.reactive(None)
model_data_state = solara.reactive(None)
agent_data_state = solara.reactive(None)
has_run_state = solara.reactive(False)


# -----------------------------
# Solara Page
# -----------------------------

@solara.component
def Page():
    solara.Title("Recommendation Backfire ABM")

    with solara.Column(gap="16px"):
        solara.Markdown(
            """
# Recommendation Backfire ABM

**Research question:**  
When do recommendation systems reduce polarization by exposing users to cross-cutting content, and when do user feedback loops cause such exposure to backfire by narrowing future recommendations and reinforcing ideological extremity?

The model represents a simplified content recommendation platform:

1. The platform recommends ideological content to each user.
2. The user accepts, ignores, or rejects the content depending on ideological distance.
3. The user's opinion may update through assimilation or backfire.
4. The recommender learns from user feedback.
5. Repeated feedback can broaden or narrow future exposure.
"""
        )

        with solara.Sidebar():
            solara.Markdown("## Model Parameters")
            solara.Markdown(
                """
Use these controls to change the simulated recommendation environment.
Each parameter changes either the user population, the recommendation algorithm,
or the psychological response rule.
"""
            )

            solara.Markdown("---")

            # -----------------------------
            # Population settings
            # -----------------------------
            solara.Markdown("### 1. Population")

            solara.Markdown(
                f"""
**Number of agents:** `{num_agents_state.value}`  
"""
            )
            solara.SliderInt(
                "Number of agents",
                value=num_agents_state,
                min=50,
                max=500,
                step=50,
            )

            solara.Markdown(
                f"""
**Simulation steps:** `{steps_state.value}`  
Number of rounds of recommendation, user response, and algorithmic updating.
"""
            )
            solara.SliderInt(
                "Steps",
                value=steps_state,
                min=25,
                max=400,
                step=25,
            )

            solara.Markdown(
                f"""
**Initial opinion distribution:** `{initial_distribution_state.value}`  
Starting ideological distribution of users.  
- `polarized`: two initial clusters  
- `uniform`: users spread across the full opinion range  
- `moderate`: users mostly start near the center
"""
            )
            solara.Select(
                label="Initial opinion distribution",
                value=initial_distribution_state,
                values=["polarized", "uniform", "moderate"],
            )

            solara.Markdown("---")

            # -----------------------------
            # Algorithm settings
            # -----------------------------
            solara.Markdown("### 2. Recommendation Algorithm")

            solara.Markdown(
                f"""
**Initial recommendation width:** `{initial_preference_width_state.value:.2f}`  
How broad the user's recommendation environment is at the beginning.  
Higher values mean more diverse or cross-cutting exposure.
Lower values mean more ideologically narrow recommendations.
"""
            )
            solara.SliderFloat(
                "Initial recommendation width",
                value=initial_preference_width_state,
                min=0.05,
                max=1.00,
                step=0.05,
            )

            solara.Markdown(
                f"""
**Local collaborative-filtering weight (social signal):** `{social_signal_weight_state.value:.2f}`  
How much the algorithm uses the recent accepted content of similar users to personalize each agent's recommendation.  
0 disables this channel; higher values let cluster-level acceptance patterns influence the content center and width.
"""
            )
            solara.SliderFloat(
                "Local collaborative-filtering weight",
                value=social_signal_weight_state,
                min=0.0,
                max=0.9,
                step=0.05,
            )

            solara.Markdown(
                f"""
**Global trending weight (cross-user signal):** `{trending_weight_state.value:.2f}`  
How much each agent's recommendation is pulled toward the population-level mean of recently accepted content (the "viral" / "trending" channel).  
0 disables this channel; higher values mean trending content is more strongly injected into every user's exposure, regardless of personalization.

**Note:** Local CF + global trending must sum to at most 1.0.
"""
            )
            solara.SliderFloat(
                "Global trending weight",
                value=trending_weight_state,
                min=0.0,
                max=0.9,
                step=0.05,
            )

            solara.Markdown("---")

            # -----------------------------
            # User psychology settings
            # -----------------------------
            solara.Markdown("### 3. User Psychology")

            solara.Markdown(
                f"""
**High-tolerance agent share:** `{high_tolerance_share_state.value:.2f}`  
Proportion of users who are more willing to accept ideologically different content.  
Higher values mean more tolerant users in the population.
"""
            )
            solara.SliderFloat(
                "High-tolerance agent share",
                value=high_tolerance_share_state,
                min=0.00,
                max=1.00,
                step=0.05,
            )

            solara.Markdown(
                f"""
**Adaptive tolerance:** `{adaptive_tolerance_state.value}`  
Whether users' tolerance thresholds can change over time.  
If enabled, successful exposure can increase tolerance, while rejection can decrease tolerance.
"""
            )
            solara.Checkbox(
                label="Adaptive tolerance",
                value=adaptive_tolerance_state,
            )

            solara.Markdown(
                f"""
**Assimilation rate:** `{assimilation_rate_state.value:.2f}`  
How strongly users move toward content they accept.  
Higher values mean accepted content changes opinions more quickly.
"""
            )
            solara.SliderFloat(
                "Assimilation rate",
                value=assimilation_rate_state,
                min=0.01,
                max=0.30,
                step=0.01,
            )

            solara.Markdown(
                f"""
**Backfire rate:** `{backfire_rate_state.value:.2f}`  
How strongly users move away from content they reject.  
Higher values mean rejected content produces stronger polarization pressure.
"""
            )
            solara.SliderFloat(
                "Backfire rate",
                value=backfire_rate_state,
                min=0.01,
                max=0.30,
                step=0.01,
            )

            solara.Markdown("---")

            # -----------------------------
            # Reproducibility
            # -----------------------------
            solara.Markdown("### 4. Reproducibility")

            solara.Markdown(
                f"""
**Random seed:** `{seed_state.value}`  
"""
            )
            solara.SliderInt(
                "Random seed",
                value=seed_state,
                min=1,
                max=999,
                step=1,
            )

            solara.Markdown("---")

            def on_run():
                # Validate constraint: social + trending must be <= 1.0.
                # If the user has pushed both sliders too high, cap trending.
                sw = social_signal_weight_state.value
                tw = trending_weight_state.value
                if sw + tw > 1.0:
                    tw = max(0.0, 1.0 - sw)
                    trending_weight_state.value = tw

                model, model_data, agent_data = run_simulation(
                    num_agents=num_agents_state.value,
                    steps=steps_state.value,
                    initial_preference_width=initial_preference_width_state.value,
                    high_tolerance_share=high_tolerance_share_state.value,
                    adaptive_tolerance=adaptive_tolerance_state.value,
                    assimilation_rate=assimilation_rate_state.value,
                    backfire_rate=backfire_rate_state.value,
                    initial_distribution=initial_distribution_state.value,
                    social_signal_weight=sw,
                    trending_weight=tw,
                    seed=seed_state.value,
                )

                model_state.value = model
                model_data_state.value = model_data
                agent_data_state.value = agent_data
                has_run_state.value = True

            solara.Button(
                "Run simulation",
                on_click=on_run,
                color="primary",
            )

        if not has_run_state.value:
            solara.Info("Set the parameters in the sidebar, then click **Run simulation**.")
            return

        model = model_state.value
        model_data = model_data_state.value

        if model is None or model_data is None:
            solara.Warning("No model data available yet.")
            return

        solara.Markdown("## Final Summary")

        SummaryGrid(model_data)

        solara.Markdown("## Process Plots")

        solara.Markdown(
            """
These plots show the main feedback process: users react to recommended content,
and the algorithm updates future exposure based on that reaction.
"""
        )

        with solara.Columns([1, 1]):
            with solara.Column():
                fig_process_1 = make_opinion_vs_algorithm_plot(model_data)
                solara.FigureMatplotlib(fig_process_1)
                solara.Markdown(
                    """
**How to read this plot:**  
This compares users' average ideological extremity with the extremity of the algorithm's learned preference profile.  
If the two lines move together, the recommender is co-evolving with user opinion rather than acting as a fixed external force.
"""
                )

            with solara.Column():
                fig_process_2 = make_feedback_rates_plot(model_data)
                solara.FigureMatplotlib(fig_process_2)
                solara.Markdown(
                    """
**How to read this plot:**  
This shows the cumulative share of recommendations that users accept, ignore, or reject.  
A rising acceptance rate with a falling backfire rate suggests that the algorithm is learning to avoid content users dislike.
"""
                )

        solara.Markdown("## Main Outcome Plots")

        with solara.Columns([1, 1]):
            with solara.Column():
                fig1 = make_line_plot(
                    model_data,
                    "Mean Extremity",
                    "Mean Extremity Over Time",
                    "Mean ideological extremity",
                )
                solara.FigureMatplotlib(fig1)
                solara.Markdown(
                    """
**How to read this plot:**  
Mean extremity measures how far users are from the ideological center on average.  
Higher values indicate stronger movement toward ideological poles.
"""
                )

            with solara.Column():
                fig4 = make_line_plot(
                    model_data,
                    "Mean Preference Width",
                    "Algorithmic Narrowing Over Time",
                    "Mean recommendation width",
                )
                solara.FigureMatplotlib(fig4)
                solara.Markdown(
                    """
**How to read this plot:**  
Preference width measures how broad or narrow the recommendation environment is.  
A falling line means the algorithm is narrowing future exposure around what users appear to prefer.
"""
                )

        # Cross-user signal diagnostics (v2)
        solara.Markdown("## Cross-User Signal Diagnostics")
        solara.Markdown(
            """
These plots make visible what the collaborative-filtering and trending channels are
actually doing across the population. They are most informative when the
**Local collaborative-filtering weight** or **Global trending weight** sliders are above zero.
"""
        )

        with solara.Columns([1, 1]):
            with solara.Column():
                fig_bimod = make_line_plot(
                    model_data,
                    "Opinion Bimodality",
                    "Opinion Bimodality Over Time",
                    "Bimodality index",
                )
                solara.FigureMatplotlib(fig_bimod)
                solara.Markdown(
                    """
**How to read this plot:**  
Bimodality measures whether opinions cluster into two (or more) groups rather than spreading evenly.  
Higher values indicate stronger emergent cluster formation.
"""
                )

            with solara.Column():
                fig_hist = make_line_plot(
                    model_data,
                    "Mean Acceptance History Filled",
                    "Acceptance History Coverage Over Time",
                    "Mean fill (0 - 1)",
                )
                solara.FigureMatplotlib(fig_hist)
                solara.Markdown(
                    """
**How to read this plot:**  
This shows how much of each agent's recent-acceptance memory is filled with data.  
This is informative when collaborative filtering or trending channels are active:
early in a run the cross-user signal is weak (memories are empty); long-running or
high-acceptance regimes have stronger collaborative signal.
"""
                )

        with solara.Columns([1, 1]):
            with solara.Column():
                fig_trend_mean = make_line_plot(
                    model_data,
                    "Trending Pool Mean",
                    "Trending Pool Mean Ideology Over Time",
                    "Trending pool mean",
                )
                solara.FigureMatplotlib(fig_trend_mean)
                solara.Markdown(
                    """
**How to read this plot:**  
The trending pool is the population-level pool of recently accepted content
that the algorithm injects into each user's recommendation when the global
trending weight is above zero. This line shows the **mean ideology** of that
pool over time: drift away from zero indicates that a dominant ideological
side is producing more of the population's accepted content. When the trending
weight is zero, this line is flat at zero (the channel is inactive).
"""
                )

            with solara.Column():
                fig_trend_std = make_line_plot(
                    model_data,
                    "Trending Pool Std",
                    "Trending Pool Ideological Spread Over Time",
                    "Trending pool standard deviation",
                )
                solara.FigureMatplotlib(fig_trend_std)
                solara.Markdown(
                    """
**How to read this plot:**  
Standard deviation of ideology within the trending pool. Larger values mean
trending content remains ideologically diverse; smaller values mean the pool
has consolidated around a narrow band of content, which can indicate cluster
lock-in. Flat at zero when the trending channel is inactive.
"""
                )

        solara.Markdown("## Final Distribution Plots")

        with solara.Columns([1, 1]):
            with solara.Column():
                fig5 = make_final_opinion_histogram(model)
                solara.FigureMatplotlib(fig5)
                solara.Markdown(
                    """
**How to read this plot:**  
This shows the final distribution of user opinions.  
Peaks near both ends suggest polarization; concentration near the center suggests moderation.
"""
                )

            with solara.Column():
                fig6 = make_preference_width_histogram(model)
                solara.FigureMatplotlib(fig6)
                solara.Markdown(
                    """
**How to read this plot:**  
This shows how broad each user's final recommendation environment is.  
Lower values mean more personalized and narrower content exposure.
"""
                )

        solara.Markdown("## Raw Model Data")

        solara.DataFrame(model_data.reset_index(), items_per_page=10)