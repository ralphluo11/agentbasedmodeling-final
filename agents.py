# agents.py

"""
Agent definitions for the recommendation-backfire ABM.

The model represents a simplified algorithmic content platform.

Each UserAgent has:
- an ideological opinion between -1 and 1
- psychological thresholds based on Social Judgment Theory
- a learned recommendation profile used by the platform

At every step, the platform recommends an ideological content item to the user.
The user then accepts, ignores, or rejects the content depending on ideological distance.
That response affects both the user's opinion and the recommender's future behavior.
"""

from collections import deque

from mesa import Agent
import numpy as np


class UserAgent(Agent):
    """
    A platform user with an ideological opinion and psychological thresholds.

    Opinion scale:
        -1 = one ideological pole
         0 = moderate / center
        +1 = opposite ideological pole

    Social Judgment Theory zones:
        distance <= acceptance_threshold:
            content is accepted; agent assimilates toward it.

        acceptance_threshold < distance < rejection_threshold:
            content is ignored or skipped; no opinion update.

        distance >= rejection_threshold:
            content is rejected; agent moves away from it.
    """

    def __init__(
        self,
        model,
        initial_opinion,
        tolerance_type="mixed",
    ):
        # Mesa 3.x automatically gives agents a unique ID and registers them with the model.
        super().__init__(model)

        self.opinion = float(initial_opinion)

        # The recommendation system's learned belief about what this user wants.
        # Initially, the platform roughly infers the user's preference from their current opinion.
        self.preference_center = float(initial_opinion)

        # Larger values mean broader recommendations.
        # Smaller values mean algorithmically narrower content exposure.
        self.preference_width = self.model.initial_preference_width

        # Assign psychological tolerance.
        # High-tolerance agents accept a wider range of disagreement.
        # Low-tolerance agents reject disagreement more easily.
        if tolerance_type == "high":
            self.acceptance_threshold = self.model.high_acceptance_threshold
            self.rejection_threshold = self.model.high_rejection_threshold

        elif tolerance_type == "low":
            self.acceptance_threshold = self.model.low_acceptance_threshold
            self.rejection_threshold = self.model.low_rejection_threshold

        else:
            # Mixed population: draw high or low tolerance probabilistically.
            if self.random.random() < self.model.high_tolerance_share:
                self.acceptance_threshold = self.model.high_acceptance_threshold
                self.rejection_threshold = self.model.high_rejection_threshold
            else:
                self.acceptance_threshold = self.model.low_acceptance_threshold
                self.rejection_threshold = self.model.low_rejection_threshold

        # Counters used for model-level measurement.
        self.accept_count = 0
        self.ignore_count = 0
        self.reject_count = 0
        self.total_exposures = 0

        # Store last-step information for debugging / visualization.
        self.last_content = None
        self.last_distance = None
        self.last_response = None

        # Collaborative filtering: record recent accepted content ideologies.
        # The platform uses this history to identify behaviorally similar users,
        # which then influences both the center and the width of future
        # recommendations (see model.recommend_content and model.find_neighbors).
        # This is the channel that makes one user's behavior affect what other
        # users see, providing the genuine agent-agent interdependence required
        # for the model to behave as an ABM rather than as N independent
        # single-user simulations.
        self.acceptance_history = deque(maxlen=self.model.acceptance_history_length)

    def step(self):
        """
        One model step for a user:
        1. receive recommended content
        2. evaluate ideological distance
        3. update opinion
        4. give behavioral feedback
        5. allow recommender profile and optional tolerance to adapt
        """

        content_ideology = self.model.recommend_content(self)
        distance = abs(content_ideology - self.opinion)

        self.last_content = content_ideology
        self.last_distance = distance
        self.total_exposures += 1

        if distance <= self.acceptance_threshold:
            response = "accept"
            self.accept_content(content_ideology)

        elif distance >= self.rejection_threshold:
            response = "reject"
            self.reject_content(content_ideology)

        else:
            response = "ignore"
            self.ignore_content(content_ideology)

        self.last_response = response

    def accept_content(self, content_ideology):
        """
        Acceptance / assimilation.

        The agent moves slightly toward the content.
        The platform interprets this as positive feedback and becomes more likely
        to recommend similar content in the future.
        """

        self.accept_count += 1

        # Record the accepted content for collaborative-filtering similarity.
        self.acceptance_history.append(float(content_ideology))

        # Opinion assimilation.
        self.opinion += self.model.assimilation_rate * (
            content_ideology - self.opinion
        )
        self.opinion = self.clip_opinion(self.opinion)

        # Positive feedback:
        # recommender shifts toward accepted content.
        self.preference_center += self.model.feedback_sensitivity * (
            content_ideology - self.preference_center
        )
        self.preference_center = self.clip_opinion(self.preference_center)

        # Accepted content may slightly narrow the recommender around revealed preference.
        self.preference_width *= self.model.accept_width_multiplier
        self.preference_width = self.clip_width(self.preference_width)

        # Optional adaptive tolerance:
        # successful exposure can make the user slightly more open to future difference.
        if self.model.adaptive_tolerance:
            self.acceptance_threshold += self.model.tolerance_learning_rate
            self.rejection_threshold += self.model.tolerance_learning_rate
            self.clip_thresholds()

    def ignore_content(self, content_ideology):
        """
        Noncommitment / skip.

        The agent does not update their opinion.
        The platform receives weak negative or neutral feedback.
        """

        self.ignore_count += 1

        # Weak feedback:
        # the algorithm may very slightly narrow exposure,
        # but much less than after explicit rejection.
        self.preference_width *= self.model.ignore_width_multiplier
        self.preference_width = self.clip_width(self.preference_width)

        # For simplicity, ignoring content does not change psychological tolerance.
        # This corresponds to the Social Judgment Theory noncommitment zone.
        if self.model.adaptive_tolerance and self.model.ignore_changes_tolerance:
            self.acceptance_threshold += self.model.ignore_tolerance_change
            self.rejection_threshold += self.model.ignore_tolerance_change
            self.clip_thresholds()

    def reject_content(self, content_ideology):
        """
        Rejection / backfire.

        The agent moves away from the content.
        The platform interprets rejection as negative feedback and reduces
        future exposure to similar content.

        This is the key feedback loop:
            opposing content -> rejection -> algorithm narrows exposure
            -> fewer future cross-cutting recommendations.
        """

        self.reject_count += 1

        # Psychological backfire:
        # move away from content. Direction is based on whether content is to the
        # left or right of the agent's current opinion.
        if content_ideology > self.opinion:
            self.opinion -= self.model.backfire_rate * abs(
                content_ideology - self.opinion
            )
        else:
            self.opinion += self.model.backfire_rate * abs(
                content_ideology - self.opinion
            )

        self.opinion = self.clip_opinion(self.opinion)

        # Negative feedback:
        # the recommender retreats toward the user's current opinion and narrows.
        # This abstracts from proprietary algorithms but captures the idea:
        # "I dislike this" makes similar content less likely in the future.
        self.preference_center += self.model.feedback_sensitivity * (
            self.opinion - self.preference_center
        )
        self.preference_center = self.clip_opinion(self.preference_center)

        self.preference_width *= self.model.reject_width_multiplier
        self.preference_width = self.clip_width(self.preference_width)

        # Optional adaptive tolerance:
        # rejection can make the user more defensive in the future.
        if self.model.adaptive_tolerance:
            self.acceptance_threshold -= self.model.defensive_rate
            self.rejection_threshold -= self.model.defensive_rate
            self.clip_thresholds()

    def clip_opinion(self, value):
        """Keep ideological values within [-1, 1]."""
        return float(np.clip(value, -1.0, 1.0))

    def clip_width(self, value):
        """Keep recommendation width within a reasonable range."""
        return float(
            np.clip(
                value,
                self.model.min_preference_width,
                self.model.max_preference_width,
            )
        )

    def clip_thresholds(self):
        """
        Keep thresholds valid and ordered.

        acceptance_threshold must remain lower than rejection_threshold.
        """

        self.acceptance_threshold = float(
            np.clip(
                self.acceptance_threshold,
                self.model.min_acceptance_threshold,
                self.model.max_acceptance_threshold,
            )
        )

        self.rejection_threshold = float(
            np.clip(
                self.rejection_threshold,
                self.model.min_rejection_threshold,
                self.model.max_rejection_threshold,
            )
        )

        # Ensure rejection threshold is always above acceptance threshold.
        if self.rejection_threshold <= self.acceptance_threshold:
            self.rejection_threshold = self.acceptance_threshold + 0.05

        self.rejection_threshold = float(
            np.clip(
                self.rejection_threshold,
                self.model.min_rejection_threshold,
                self.model.max_rejection_threshold,
            )
        )