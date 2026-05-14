# model.py

"""
This model simulates a population of agents with ideological opinions who receive content recommendations from an algorithm. The algorithm learns from user feedback, and users update their opinions based on the
content they see and their psychological tolerance for disagreement. The model captures the feedback loop between user behavior and algorithmic recommendations, which can lead to polarization or moderation depending on the parameters.
"""

from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np

from agents import UserAgent


def get_agents(model):
    """
    Mesa 3.x stores agents in model.agents.
    Convert to list so Python and NumPy operations work normally.
    """
    return list(model.agents)



# Model-level reporters to compute aggregate metrics at each step.

def mean_opinion(model):
    opinions = [agent.opinion for agent in get_agents(model)]
    return float(np.mean(opinions))


def mean_abs_opinion(model):
    """
    Average absolute distance from the ideological center.
    Higher values indicate more ideological extremity.
    """
    opinions = [agent.opinion for agent in get_agents(model)]
    return float(np.mean(np.abs(opinions)))


def opinion_variance(model):
    opinions = [agent.opinion for agent in get_agents(model)]
    return float(np.var(opinions))


def extreme_share(model):
    """
    Share of agents near ideological extremes.
    """
    opinions = [agent.opinion for agent in get_agents(model)]
    if len(opinions) == 0:
        return 0.0
    return float(np.mean([abs(x) >= model.extreme_cutoff for x in opinions]))


def mean_preference_center(model):
    """
    Average algorithmic estimate of users' preferred content.
    """
    centers = [agent.preference_center for agent in get_agents(model)]
    return float(np.mean(centers))


def mean_abs_preference_center(model):
    """
    Average extremity of the algorithm's learned preference centers.
    """
    centers = [agent.preference_center for agent in get_agents(model)]
    return float(np.mean(np.abs(centers)))


def mean_preference_width(model):
    """
    Average breadth of algorithmic recommendation profiles.
    Lower values mean narrower content diets.
    """
    widths = [agent.preference_width for agent in get_agents(model)]
    return float(np.mean(widths))


def mean_acceptance_threshold(model):
    thresholds = [agent.acceptance_threshold for agent in get_agents(model)]
    return float(np.mean(thresholds))


def mean_rejection_threshold(model):
    thresholds = [agent.rejection_threshold for agent in get_agents(model)]
    return float(np.mean(thresholds))


def model_acceptance_rate(model):
    """
    Share of all exposures that produced acceptance/assimilation.
    """
    total_accepts = sum(agent.accept_count for agent in get_agents(model))
    total_exposures = sum(agent.total_exposures for agent in get_agents(model))

    if total_exposures == 0:
        return 0.0

    return float(total_accepts / total_exposures)


def model_ignore_rate(model):
    """
    Share of all exposures that produced ignore/skip.
    """
    total_ignores = sum(agent.ignore_count for agent in get_agents(model))
    total_exposures = sum(agent.total_exposures for agent in get_agents(model))

    if total_exposures == 0:
        return 0.0

    return float(total_ignores / total_exposures)


def model_backfire_rate(model):
    """
    Share of all exposures that produced rejection/backfire.

    """
    total_rejects = sum(agent.reject_count for agent in get_agents(model))
    total_exposures = sum(agent.total_exposures for agent in get_agents(model))

    if total_exposures == 0:
        return 0.0

    return float(total_rejects / total_exposures)


def average_exposure_distance(model):
    """
    Average ideological distance between agents and their most recently
    recommended content.
    """
    distances = [
        agent.last_distance
        for agent in get_agents(model)
        if agent.last_distance is not None
    ]

    if len(distances) == 0:
        return 0.0

    return float(np.mean(distances))


# -----------------------------
# Main model
# -----------------------------

class RecommendationBackfireModel(Model):
    """
    Main model class for Mesa 3.x.
    """

    def __init__(
        self,
        num_agents=200,
        initial_distribution="polarized",
        high_tolerance_share=0.5,
        feedback_sensitivity=0.25,
        initial_preference_width=0.45,
        adaptive_tolerance=True,
        assimilation_rate=0.08,
        backfire_rate=0.06,
        seed=None,
    ):
        super().__init__(seed=seed)

        # Core parameters
        self.num_agents = num_agents
        self.initial_distribution = initial_distribution
        self.high_tolerance_share = high_tolerance_share
        self.feedback_sensitivity = feedback_sensitivity
        self.initial_preference_width = initial_preference_width
        self.adaptive_tolerance = adaptive_tolerance
        self.assimilation_rate = assimilation_rate

        # This is the strength of opinion movement after rejection.
        self.backfire_rate = backfire_rate

        # Psychological threshold parameters.
        # High-tolerance agents accept more disagreement and reject only distant content.
        self.high_acceptance_threshold = 0.35
        self.high_rejection_threshold = 0.85

        # Low-tolerance agents have a narrow acceptance zone and reject more easily.
        self.low_acceptance_threshold = 0.18
        self.low_rejection_threshold = 0.55

        # Bounds for adaptive tolerance.
        self.min_acceptance_threshold = 0.05
        self.max_acceptance_threshold = 0.60
        self.min_rejection_threshold = 0.25
        self.max_rejection_threshold = 1.20

        # Adaptive tolerance update rates.
        self.tolerance_learning_rate = 0.005
        self.defensive_rate = 0.008

        # Whether to ignore changes in tolerance when updating based on feedback.
        self.ignore_changes_tolerance = False
        self.ignore_tolerance_change = 0.0

        # Recommendation width parameters.
        self.min_preference_width = 0.05
        self.max_preference_width = 1.00

        # Width multipliers:
        # accepted content slightly narrows the profile around successful content.
        # ignored content very slightly narrows.
        # rejected content narrows much more strongly.
        self.accept_width_multiplier = 0.995
        self.ignore_width_multiplier = 0.998
        self.reject_width_multiplier = 0.94

        # Measurement.
        self.extreme_cutoff = 0.75

        # Create agents before collecting data.
        self.create_agents()

        self.datacollector = DataCollector(
            model_reporters={
                "Mean Opinion": mean_opinion,
                "Mean Extremity": mean_abs_opinion,
                "Opinion Variance": opinion_variance,
                "Extreme Share": extreme_share,
                "Mean Preference Center": mean_preference_center,
                "Mean Preference Center Extremity": mean_abs_preference_center,
                "Mean Preference Width": mean_preference_width,
                "Mean Acceptance Threshold": mean_acceptance_threshold,
                "Mean Rejection Threshold": mean_rejection_threshold,
                "Acceptance Rate": model_acceptance_rate,
                "Ignore Rate": model_ignore_rate,
                "Backfire Rate": model_backfire_rate,
                "Average Exposure Distance": average_exposure_distance,
            },
            agent_reporters={
                "Opinion": "opinion",
                "Preference Center": "preference_center",
                "Preference Width": "preference_width",
                "Acceptance Threshold": "acceptance_threshold",
                "Rejection Threshold": "rejection_threshold",
                "Last Content": "last_content",
                "Last Distance": "last_distance",
                "Last Response": "last_response",
            },
        )

        self.datacollector.collect(self)

    def create_agents(self):
        """
        Create agents with initial ideological opinions.
        """

        for _ in range(self.num_agents):
            initial_opinion = self.draw_initial_opinion()

            UserAgent(
                model=self,
                initial_opinion=initial_opinion,
                tolerance_type="mixed",
            )

    def draw_initial_opinion(self):
        """
        Draw an initial ideological opinion.

        Initial distributions:
        - polarized: two clusters around -0.45 and +0.45
        - uniform: random opinions across [-1, 1]
        - moderate: mostly centered around 0
        """

        if self.initial_distribution == "uniform":
            opinion = self.random.uniform(-1.0, 1.0)

        elif self.initial_distribution == "moderate":
            opinion = np.random.normal(loc=0.0, scale=0.25)

        else:
            # default is slightly polarized distribution with two clusters around -0.45 and +0.45
            if self.random.random() < 0.5:
                opinion = np.random.normal(loc=-0.45, scale=0.18)
            else:
                opinion = np.random.normal(loc=0.45, scale=0.18)

        return float(np.clip(opinion, -1.0, 1.0))

    def recommend_content(self, agent):
        """
        Recommend a content ideology to an agent.

        The recommender samples content from a normal distribution centered on
        the platform's current learned preference_center for that user.
        """

        content = np.random.normal(
            loc=agent.preference_center,
            scale=agent.preference_width,
        )

        return float(np.clip(content, -1.0, 1.0))

    def step(self):
        """
        Run one step of the model: each agent receives a recommendation and updates.
        """
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    def run_model(self, n_steps=100):
        """
        Run the model for n_steps.
        """
        for _ in range(n_steps):
            self.step()