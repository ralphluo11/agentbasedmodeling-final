# model.py

"""
Recommendation-backfire ABM.

This model asks when algorithmically introduced cross-cutting exposure reduces
polarization, and when user rejection teaches the recommender system to narrow
future recommendations.

Core mechanism:

    content recommendation
        -> psychological response
        -> user feedback
        -> algorithmic update
        -> future recommendation
        -> aggregate polarization or moderation

This version is written for Mesa 3.x.
"""

from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np

from agents import UserAgent


# -----------------------------
# Model-level reporters
# -----------------------------

def mean_opinion(model):
    return float(np.mean([a.opinion for a in model.agents]))


def mean_abs_opinion(model):
    """
    Average absolute distance from the ideological center.
    Higher values indicate more ideological extremity.
    """
    return float(np.mean([abs(a.opinion) for a in model.agents]))


def opinion_variance(model):
    return float(np.var([a.opinion for a in model.agents]))


def extreme_share(model):
    """
    Share of agents near ideological extremes.
    """
    if len(model.agents) == 0:
        return 0.0
    return float(np.mean(
        [abs(a.opinion) >= model.extreme_cutoff for a in model.agents]
    ))


def mean_preference_center(model):
    """
    Average algorithmic estimate of users' preferred content.
    """
    return float(np.mean([a.preference_center for a in model.agents]))


def mean_abs_preference_center(model):
    """
    Average extremity of the algorithm's learned preference centers.
    """
    return float(np.mean([abs(a.preference_center) for a in model.agents]))


def mean_preference_width(model):
    """
    Average breadth of algorithmic recommendation profiles.
    Lower values mean narrower content diets.
    """
    return float(np.mean([a.preference_width for a in model.agents]))


def mean_acceptance_threshold(model):
    return float(np.mean([a.acceptance_threshold for a in model.agents]))


def mean_rejection_threshold(model):
    return float(np.mean([a.rejection_threshold for a in model.agents]))


def model_acceptance_rate(model):
    """
    Share of all exposures that produced acceptance/assimilation.
    """
    total_accepts = sum(a.accept_count for a in model.agents)
    total_exposures = sum(a.total_exposures for a in model.agents)
    if total_exposures == 0:
        return 0.0
    return float(total_accepts / total_exposures)


def model_ignore_rate(model):
    """
    Share of all exposures that produced ignore/skip.
    """
    total_ignores = sum(a.ignore_count for a in model.agents)
    total_exposures = sum(a.total_exposures for a in model.agents)
    if total_exposures == 0:
        return 0.0
    return float(total_ignores / total_exposures)


def model_backfire_rate(model):
    """
    Share of all exposures that produced rejection/backfire.

    Important: this function is named model_backfire_rate to avoid conflict
    with the parameter self.backfire_rate, which is the strength of opinion
    movement after rejection.
    """
    total_rejects = sum(a.reject_count for a in model.agents)
    total_exposures = sum(a.total_exposures for a in model.agents)
    if total_exposures == 0:
        return 0.0
    return float(total_rejects / total_exposures)


def average_exposure_distance(model):
    """
    Average ideological distance between agents and their most recently
    recommended content.
    """
    distances = [a.last_distance for a in model.agents
                 if a.last_distance is not None]
    if len(distances) == 0:
        return 0.0
    return float(np.mean(distances))


def mean_acceptance_history_length(model):
    """
    Average filled length of agents' acceptance histories, normalized to [0,1].

    This is a population-level diagnostic for the collaborative-filtering
    channel: it indicates how informative the cluster-similarity signal is
    at a given step. Early in a run this will be near zero; in long-running
    or high-acceptance regimes it approaches 1.
    """
    if len(model.agents) == 0:
        return 0.0
    return float(np.mean([
        len(a.acceptance_history) / model.acceptance_history_length
        for a in model.agents
    ]))


def opinion_bimodality(model):
    """
    A simple bimodality diagnostic: variance of opinions minus variance of a
    Gaussian with the same range. Higher values indicate clustering away from
    a single-mode distribution. This complements opinion_variance for
    detecting emergent multi-cluster structure under collaborative filtering.
    """
    opinions = np.array([a.opinion for a in model.agents])
    if len(opinions) < 2:
        return 0.0
    median = float(np.median(opinions))
    left = opinions[opinions < median]
    right = opinions[opinions >= median]
    if len(left) < 2 or len(right) < 2:
        return 0.0
    between_gap = abs(float(np.mean(right)) - float(np.mean(left)))
    within_spread = float(np.std(left)) + float(np.std(right))
    if within_spread < 1e-9:
        return 0.0
    return float(between_gap / within_spread)


def trending_pool_mean(model):
    """
    Mean ideology of the global trending pool at the current step.

    Diagnostic for the cross-user trending channel: shows what aggregate
    ideological signal the algorithm is currently injecting into all users'
    exposure distributions. Returns 0.0 when the trending channel is inactive
    or the pool is empty.
    """
    if not model.trending_pool:
        return 0.0
    return float(np.mean(model.trending_pool))


def trending_pool_std(model):
    """
    Ideological spread (standard deviation) of the global trending pool.

    Together with trending_pool_mean, this characterizes the population-level
    cross-user signal: a small std means the trending pool has consolidated
    around a narrow band of content (potential cluster lock-in), while a
    larger std means trending content remains ideologically diverse.
    """
    if not model.trending_pool or len(model.trending_pool) < 2:
        return 0.0
    return float(np.std(model.trending_pool))


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
        social_signal_weight=0.0,
        trending_weight=0.0,
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
        self.backfire_rate = backfire_rate

        # Cross-user signaling weights.
        # social_signal_weight: local collaborative-filtering strength.
        # trending_weight: global ideology-agnostic viral channel strength.
        # Together they must sum to <= 1; when both are 0 the model reduces
        # to the individual-learning baseline.
        self.social_signal_weight = social_signal_weight
        self.trending_weight = trending_weight
        if self.social_signal_weight + self.trending_weight > 1.0:
            raise ValueError(
                "social_signal_weight + trending_weight must be <= 1.0; "
                f"got {self.social_signal_weight} + {self.trending_weight}"
            )

        # Collaborative-filtering structural parameters (held fixed across
        # experiments; the substantive variation is captured by
        # social_signal_weight).
        self.k_neighbors = 5
        self.acceptance_history_length = 10

        # Trending pool: refreshed at the end of each step by concatenating
        # all agents' acceptance histories across the population.
        self.trending_pool_window = 5
        self.trending_pool = []

        # Cluster-level width modulation: high cluster acceptance tightens
        # widths in synchrony, low cluster acceptance loosens them.
        self.cluster_high_accept_threshold = 0.60
        self.cluster_low_accept_threshold = 0.30
        self.cluster_tighten_multiplier = 0.90
        self.cluster_loosen_multiplier = 1.10

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

        # Usually leave this False for theoretical clarity:
        # noncommitment means no psychological update.
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

        # -----------------------------
        # Create agents inline.
        # -----------------------------
        # Initial opinion is drawn from one of three distributions:
        #   polarized: two clusters around -0.45 and +0.45
        #   uniform:   random opinions across [-1, 1]
        #   moderate:  mostly centered around 0
        # Each agent is registered with the model automatically by Mesa 3.x
        # when UserAgent is instantiated with model=self.
        for _ in range(self.num_agents):
            if self.initial_distribution == "uniform":
                opinion = self.random.uniform(-1.0, 1.0)
            elif self.initial_distribution == "moderate":
                opinion = np.random.normal(loc=0.0, scale=0.25)
            else:
                # Default: mildly polarized population.
                # Half starts left-of-center, half starts right-of-center.
                if self.random.random() < 0.5:
                    opinion = np.random.normal(loc=-0.45, scale=0.18)
                else:
                    opinion = np.random.normal(loc=0.45, scale=0.18)
            opinion = float(np.clip(opinion, -1.0, 1.0))

            UserAgent(
                model=self,
                initial_opinion=opinion,
                tolerance_type="mixed",
            )

        # -----------------------------
        # Data collection.
        # -----------------------------
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
                "Mean Acceptance History Filled": mean_acceptance_history_length,
                "Opinion Bimodality": opinion_bimodality,
                "Trending Pool Mean": trending_pool_mean,
                "Trending Pool Std": trending_pool_std,
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

    def find_neighbors(self, agent):
        """
        Find the K agents most behaviorally similar to the given agent,
        measured by their recent acceptance histories.

        Similarity is computed as the negative absolute distance between two
        users' mean recently-accepted content (smaller distance = more similar).
        Agents that have not yet accepted any content (empty history) are
        treated as having signature = their current opinion, which is the
        platform's best initial guess at their preferences.

        This function is the channel through which one user's behavior affects
        what other users see, providing the agent-agent interdependence
        required for genuine emergent dynamics.
        """
        K = self.k_neighbors
        if len(self.agents) <= 1 or K <= 0:
            return []

        target_sig = (
            float(np.mean(agent.acceptance_history))
            if len(agent.acceptance_history) > 0
            else agent.opinion
        )

        scored = []
        for other in self.agents:
            if other is agent:
                continue
            other_sig = (
                float(np.mean(other.acceptance_history))
                if len(other.acceptance_history) > 0
                else other.opinion
            )
            similarity = -abs(target_sig - other_sig)
            scored.append((similarity, other))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [other for (_, other) in scored[:K]]

    def recommend_content(self, agent):
        """
        Recommend a content ideology to an agent.

        The recommender constructs a Gaussian distribution whose center and
        width depend on three channels:

        1. Individual preference (always present, weight = 1 - w_social - w_trending).
           Uses the agent's own preference_center and preference_width, updated
           by their own past responses.

        2. Local collaborative filtering (weight = w_social).
           The K most behaviorally similar agents (similar acceptance histories)
           contribute their mean recently-accepted content to the center, and
           their aggregate acceptance rate modulates the width (high cluster
           acceptance tightens widths in synchrony; low cluster acceptance
           loosens them).

        3. Global trending (weight = w_trending).
           The same pool of recently-accepted content from the population is
           sampled for every agent regardless of similarity or ideology.

        The three weights sum to 1. When w_social = w_trending = 0, the model
        reduces to the individual-learning baseline.
        """
        w_social = self.social_signal_weight
        w_trending = self.trending_weight
        w_individual = 1.0 - w_social - w_trending

        # Channel 1: individual.
        individual_center = agent.preference_center
        individual_width = agent.preference_width

        # Channel 2: local collaborative filtering.
        cluster_center = individual_center
        cluster_width_factor = 1.0

        if w_social > 0:
            neighbors = self.find_neighbors(agent)
            if len(neighbors) > 0:
                neighbor_accepts = []
                for other in neighbors:
                    if len(other.acceptance_history) > 0:
                        neighbor_accepts.extend(other.acceptance_history)

                if len(neighbor_accepts) > 0:
                    cluster_center = float(np.mean(neighbor_accepts))
                else:
                    cluster_center = float(np.mean([o.opinion for o in neighbors]))

                neighbor_accept_rates = [
                    len(o.acceptance_history) / self.acceptance_history_length
                    for o in neighbors
                ]
                cluster_accept_rate = float(np.mean(neighbor_accept_rates))

                if cluster_accept_rate > self.cluster_high_accept_threshold:
                    cluster_width_factor = self.cluster_tighten_multiplier
                elif cluster_accept_rate < self.cluster_low_accept_threshold:
                    cluster_width_factor = self.cluster_loosen_multiplier

        # Channel 3: global trending.
        # If the pool is non-empty, the trending center is the mean of the
        # global pool. If empty (e.g. step 0), fall back to the individual channel.
        if w_trending > 0 and len(self.trending_pool) > 0:
            trending_center = float(np.mean(self.trending_pool))
        else:
            trending_center = individual_center

        # Blend the three channels.
        content_center = (
            w_individual * individual_center
            + w_social * cluster_center
            + w_trending * trending_center
        )
        # Width is modulated only by the social channel; trending uses the
        # agent's own width (the platform serves trending items at standard
        # personalization breadth, not at cluster-modulated breadth).
        content_width = individual_width * (
            (1.0 - w_social) * 1.0 + w_social * cluster_width_factor
        )

        # Clip width to allowed range before sampling.
        content_width = float(
            np.clip(content_width, self.min_preference_width, self.max_preference_width)
        )

        content = np.random.normal(loc=content_center, scale=content_width)
        return float(np.clip(content, -1.0, 1.0))

    def update_trending_pool(self):
        """
        Refresh the global trending pool with all acceptance events held in
        agents' acceptance_history deques. Called at the end of each step.
        """
        if self.trending_weight <= 0:
            # Skip computation when trending channel is disabled.
            self.trending_pool = []
            return

        pool = []
        for agent in self.agents:
            if len(agent.acceptance_history) > 0:
                pool.extend(list(agent.acceptance_history))
        self.trending_pool = pool

    def step(self):
        """
        Mesa 3.x replacement for RandomActivation:
        shuffle agents, then call each agent's step method.
        """
        self.agents.shuffle_do("step")
        self.update_trending_pool()
        self.datacollector.collect(self)

    def run_model(self, n_steps=100):
        """
        Run the model for n_steps.
        """
        for _ in range(n_steps):
            self.step()
