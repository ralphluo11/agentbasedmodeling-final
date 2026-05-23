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
# Helper
# -----------------------------

def get_agents(model):
    """
    Mesa 3.x stores agents in model.agents.
    Convert to list so Python and NumPy operations work normally.
    """
    return list(model.agents)


# -----------------------------
# Model-level reporters
# -----------------------------

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

    Important: this function is named model_backfire_rate to avoid conflict
    with the parameter self.backfire_rate, which is the strength of opinion
    movement after rejection.
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


def mean_acceptance_history_length(model):
    """
    Average filled length of agents' acceptance histories, normalized to [0,1].

    This is a population-level diagnostic for the collaborative-filtering
    channel: it indicates how informative the cluster-similarity signal is
    at a given step. Early in a run this will be near zero; in long-running
    or high-acceptance regimes it approaches 1.
    """
    if len(get_agents(model)) == 0:
        return 0.0
    lengths = [
        len(agent.acceptance_history) / model.acceptance_history_length
        for agent in get_agents(model)
    ]
    return float(np.mean(lengths))


def opinion_bimodality(model):
    """
    A simple bimodality diagnostic: variance of opinions minus variance of a
    Gaussian with the same range. Higher values indicate clustering away from
    a single-mode distribution. This complements opinion_variance for
    detecting emergent multi-cluster structure under collaborative filtering.
    """
    opinions = np.array([agent.opinion for agent in get_agents(model)])
    if len(opinions) < 2:
        return 0.0
    # Hartigan-style heuristic: compare empirical variance to gap between two halves.
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

        # Collaborative-filtering strength.
        # 0.0 = each agent's recommendations depend only on their own history
        #       (the individual-learning baseline used in v1 of this model).
        # 1.0 = recommendations depend only on the aggregate behavior of the
        #       K most behaviorally similar agents, with no individual signal.
        # Intermediate values mix the two. This parameter is the central new
        # axis introduced in v2 and is swept as an experimental dimension.
        self.social_signal_weight = social_signal_weight

        # Global-trending strength (ideology-agnostic viral channel).
        # 0.0 = no trending injection.
        # > 0  = a fraction of each recommendation is drawn from a population-
        #        wide pool of recently-accepted content. Unlike the local
        #        collaborative-filtering channel, this pool is the SAME for
        #        every agent regardless of their own opinion or cluster: it
        #        operationalizes the "viral content" / "trending topic" channel
        #        on real platforms, which pushes the same widely-engaged
        #        content to everyone independent of personalization.
        # Together, social_signal_weight + trending_weight must be <= 1.
        self.trending_weight = trending_weight

        # Constraint check.
        if self.social_signal_weight + self.trending_weight > 1.0:
            raise ValueError(
                "social_signal_weight + trending_weight must be <= 1.0; "
                f"got {self.social_signal_weight} + {self.trending_weight}"
            )

        # Collaborative-filtering structural parameters (held fixed across
        # experiments; the substantive variation is captured by
        # social_signal_weight above).
        self.k_neighbors = 5
        self.acceptance_history_length = 10

        # Trending-pool structural parameters.
        # The pool stores all acceptance events within the last
        # trending_pool_window steps across the population, and the algorithm
        # samples uniformly from this pool when injecting trending content.
        self.trending_pool_window = 5
        # Initialized as an empty list; populated at the end of each step.
        self.trending_pool = []

        # Cluster-level width modulation.
        # When the K-nearest neighborhood has high recent acceptance, the
        # algorithm interprets this as evidence that members of this cluster
        # prefer narrower content; preference widths tighten. When neighborhood
        # acceptance is low, widths loosen. These are cluster-level effects:
        # similar agents experience parallel changes even when their own
        # individual reactions differ.
        self.cluster_high_accept_threshold = 0.60
        self.cluster_low_accept_threshold = 0.30
        self.cluster_tighten_multiplier = 0.90
        self.cluster_loosen_multiplier = 1.10

        # This is the strength of opinion movement after rejection.
        # It is not the same as the measured "Backfire Rate."
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

    def create_agents(self):
        """
        Create agents with initial ideological opinions.

        In Mesa 3.x, agents automatically register with the model when created.
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
            # Default: mildly polarized population.
            # Half starts left-of-center, half starts right-of-center.
            if self.random.random() < 0.5:
                opinion = np.random.normal(loc=-0.45, scale=0.18)
            else:
                opinion = np.random.normal(loc=0.45, scale=0.18)

        return float(np.clip(opinion, -1.0, 1.0))

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

        all_agents = self.agents
        K = self.k_neighbors

        if len(all_agents) <= 1 or K <= 0:
            return []

        target_sig = (
            float(np.mean(agent.acceptance_history))
            if len(agent.acceptance_history) > 0
            else agent.opinion
        )

        # Score every other agent by similarity (negative distance).
        scored = []
        for other in all_agents:
            if other is agent:
                continue
            other_sig = (
                float(np.mean(other.acceptance_history))
                if len(other.acceptance_history) > 0
                else other.opinion
            )
            similarity = -abs(target_sig - other_sig)
            scored.append((similarity, other))

        # Top K by similarity (highest = closest signature).
        scored.sort(key=lambda x: x[0], reverse=True)
        return [other for (_, other) in scored[:K]]

    def recommend_content(self, agent):
        """
        Recommend a content ideology to an agent.

        The recommender constructs a Gaussian distribution whose center and
        width depend on three channels:

        1. Individual preference (always present, weight = 1 - w_social - w_trending).
           Uses the agent's own preference_center and preference_width, updated
           by their own past responses. Equivalent to the v1 model.

        2. Local collaborative filtering (weight = w_social).
           The K most behaviorally similar agents (similar acceptance histories)
           contribute their mean recently-accepted content to the center, and
           their aggregate acceptance rate modulates the width (high cluster
           acceptance tightens widths in synchrony; low cluster acceptance
           loosens them). Channel 2 is what makes the model "non-reducible"
           to N independent single-user simulations: one user's behavior
           influences what behaviorally similar users see.

        3. Global trending (weight = w_trending).
           The same pool of recently-accepted content from the population is
           sampled for every agent regardless of similarity or ideology. This
           operationalizes the "viral" or "trending topic" channel on real
           platforms, which pushes widely-engaged content to all users
           independent of personalization.

        The three weights sum to 1. When w_social = w_trending = 0, the model
        reduces to v1 individual learning.
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
        # global pool, sampled uniformly. If empty (e.g. step 0), fall back
        # to the individual channel.
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
        Refresh the global trending pool with all acceptance events from the
        last trending_pool_window steps. Called at the end of each step.
        """
        if self.trending_weight <= 0:
            # Skip computation when trending channel is disabled.
            self.trending_pool = []
            return

        pool = []
        # The acceptance_history deque already holds the most recent
        # trending_pool_window accepts in the order they happened (well,
        # all accepts up to acceptance_history_length). We use the most
        # recent items only.
        for agent in self.agents:
            if len(agent.acceptance_history) > 0:
                # Take all of agent's recent accepts (deque is already
                # capped at acceptance_history_length).
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