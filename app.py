# app.py

"""
This app allows you to change model parameters, run the simulation, and visualize model outcomes.

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


# Helper functions

def run_simulation(
    num_agents,
    steps,
    initial_preference_width,
    high_tolerance_share,
    adaptive_tolerance,
    assimilation_rate,
    backfire_rate,
    initial_distribution,
    seed,
):
    """
    Create and run one model instance with the given parameters, then return the model and collected data.
    """

    model = RecommendationBackfireModel(
        num_agents=num_agents,
        initial_distribution=initial_distribution,
        high_tolerance_share=high_tolerance_share,
        feedback_sensitivity=0.25,           # ← hardcoded here
        initial_preference_width=initial_preference_width,
        adaptive_tolerance=adaptive_tolerance,
        assimilation_rate=assimilation_rate,
        backfire_rate=backfire_rate,
        seed=seed,
    )

    model.run_model(steps)

    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    return model, model_data, agent_data


def make_line_plot(model_data, y_column, title, y_label):
    """
Create a simple line plot for a given model variable over time.
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
    Plot acceptance, ignore, and backfire rates together to show the user feedback process.
    """

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(model_data.index, model_data["Acceptance Rate"], label="Acceptance")
    ax.plot(model_data.index, model_data["Ignore Rate"], label="Ignore")
    ax.plot(model_data.index, model_data["Backfire Rate"], label="Backfire")

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative share of exposures")
    ax.set_title("Cumulative User Response Rates Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def make_opinion_vs_algorithm_plot(model_data):
    """
    Plot user opinion extremity and algorithmic preference-center extremity to show the feedback loop between user psychology and algorithmic updating.
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
    Create a histogram of final agent opinions to show the overall ideological distribution at the end of the simulation.
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


def make_threshold_scatter(model):
    """
    Scatter plot showing final opinion and final acceptance threshold for each agent. This can reveal whether users with different opinions also have different tolerance levels, which may indicate that the adaptive tolerance mechanism is producing heterogeneous outcomes.
    """

    opinions = [agent.opinion for agent in model.agents]
    thresholds = [agent.acceptance_threshold for agent in model.agents]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(opinions, thresholds, alpha=0.6)
    ax.set_xlabel("Final opinion")
    ax.set_ylabel("Final acceptance threshold")
    ax.set_title("Final Opinion and Acceptance Threshold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def make_summary_items(model_data):
    """
    Return final summary metrics as a list of display tuples.
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



# Reactive state to hold model parameters and results

num_agents_state = solara.reactive(200)
steps_state = solara.reactive(150)
initial_preference_width_state = solara.reactive(0.45)
high_tolerance_share_state = solara.reactive(0.50)
adaptive_tolerance_state = solara.reactive(True)
assimilation_rate_state = solara.reactive(0.08)
backfire_rate_state = solara.reactive(0.06)
initial_distribution_state = solara.reactive("polarized")
seed_state = solara.reactive(42)

model_state = solara.reactive(None)
model_data_state = solara.reactive(None)
agent_data_state = solara.reactive(None)
has_run_state = solara.reactive(False)


# Solara Page component and section layout


@solara.component
def Page():
    solara.Title("Recommendation Backfire ABM")

    with solara.Column(gap="16px"):
        solara.Markdown(
            """
# Recommendation Backfire ABM
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
                model, model_data, agent_data = run_simulation(
                    num_agents=num_agents_state.value,
                    steps=steps_state.value,
                    initial_preference_width=initial_preference_width_state.value,
                    high_tolerance_share=high_tolerance_share_state.value,
                    adaptive_tolerance=adaptive_tolerance_state.value,
                    assimilation_rate=assimilation_rate_state.value,
                    backfire_rate=backfire_rate_state.value,
                    initial_distribution=initial_distribution_state.value,
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
                fig2 = make_line_plot(
                    model_data,
                    "Opinion Variance",
                    "Opinion Variance Over Time",
                    "Opinion variance",
                )
                solara.FigureMatplotlib(fig2)
                solara.Markdown(
                    """
**How to read this plot:**  
Opinion variance measures how dispersed the population is.  
Higher variance means users are spread farther apart ideologically.
"""
                )

        with solara.Columns([1, 1]):
            with solara.Column():
                fig3 = make_line_plot(
                    model_data,
                    "Backfire Rate",
                    "Cumulative Backfire Rate Over Time",
                    "Cumulative backfire rate",
                )
                solara.FigureMatplotlib(fig3)
                solara.Markdown(
                    """
**How to read this plot:**  
Backfire rate measures how often recommendations fall into users' rejection zones.  
If this falls over time, the platform is reducing exposure to content users reject.
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

        solara.Markdown("## Adaptive Tolerance Visualization")

        fig7 = make_threshold_scatter(model)
        solara.FigureMatplotlib(fig7)
        solara.Markdown(
            """
        **How to read this plot:**  
        This shows whether users with different final opinions also have different acceptance thresholds.  
        If many points cluster near the upper or lower threshold limits, the adaptive tolerance rule may be too strong and should be adjusted.

        **Note:** If adaptive tolerance is OFF, all agents retain their initial thresholds (0.18 for low-tolerance, 0.35 for high-tolerance) and the scatter will show only two horizontal lines—this is expected.
        """
        )
        solara.Markdown("## Raw Model Data")

        solara.DataFrame(model_data.reset_index(), items_per_page=10)