# agents.py

"""
Agent definitions for the recommendation-backfire ABM.

The model represents a simplified algorithmic content platform (tike YouTube, Facebook, Twitter, etc.) and its users.

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
        super().__init__(model)

        self.opinion = float(initial_opinion)

        # The recommendation system's learned belief about what this user wants.
        # Initially, the platform infers the user's preference directly from their current opinion.
        self.preference_center = float(initial_opinion)

        # Larger values mean broader recommendations.
        # Smaller values mean narrower content exposure.
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

 
        self.accept_count = 0
        self.ignore_count = 0
        self.reject_count = 0
        self.total_exposures = 0

        # Store last recommended content and response for data analysis and visualization.
        self.last_content = None
        self.last_distance = None
        self.last_response = None

        # Collaborative filtering: record recent accepted content ideologies.
        # The platform uses this history to identify behaviorally similar users,
        # which then influences both the center and the width of future
        # recommendations (see model.recommend_content).

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
        Acceptance 
        The agent moves slightly toward the content.
        The platform interprets this as positive feedback and becomes more likely
        to recommend similar content in the future.
        """

        self.accept_count += 1

        # Record the accepted content for collaborative-filtering similarity.
        self.acceptance_history.append(float(content_ideology))

        # Opinion assimilation.
        self.opinion += self.model.assimilation_rate * (content_ideology - self.opinion)
        self.opinion = float(np.clip(self.opinion, -1.0, 1.0))

        # Positive feedback: recommender shifts toward accepted content.
        self.preference_center += self.model.feedback_sensitivity * (
            content_ideology - self.preference_center
        )
        self.preference_center = float(np.clip(self.preference_center, -1.0, 1.0))

        # Accepted content may slightly narrow the the recommender 
        self.preference_width = float(np.clip(
            self.preference_width * self.model.accept_width_multiplier,
            self.model.min_preference_width,
            self.model.max_preference_width,
        ))

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

        # Weak feedback: the algorithm may very slightly narrow exposure,
        # but much less than after explicit rejection.
        self.preference_width = float(np.clip(
            self.preference_width * self.model.ignore_width_multiplier,
            self.model.min_preference_width,
            self.model.max_preference_width,
        ))

        # For simplicity, ignoring content does not change psychological tolerance.
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

        # Psychological backfire: move away from content. Direction is based on
        # whether content is to the left or right of the agent's current opinion.
        if content_ideology > self.opinion:
            self.opinion -= self.model.backfire_rate * abs(content_ideology - self.opinion)
        else:
            self.opinion += self.model.backfire_rate * abs(content_ideology - self.opinion)

        self.opinion = float(np.clip(self.opinion, -1.0, 1.0))

        # Negative feedback: the recommender retreats toward the user's current
        # opinion and narrows. This abstracts the idea: "I dislike this" makes similar content less likely.
        self.preference_center += self.model.feedback_sensitivity * (
            self.opinion - self.preference_center
        )
        self.preference_center = float(np.clip(self.preference_center, -1.0, 1.0))

        self.preference_width = float(np.clip(
            self.preference_width * self.model.reject_width_multiplier,
            self.model.min_preference_width,
            self.model.max_preference_width,
        ))

        # Optional adaptive tolerance:
        # rejection can make the user more defensive in the future.
        if self.model.adaptive_tolerance:
            self.acceptance_threshold -= self.model.defensive_rate
            self.rejection_threshold -= self.model.defensive_rate
            self.clip_thresholds()

    def clip_thresholds(self):
        """
        Keep thresholds valid and ordered to ensure the model behaves as intended.

        Maintains two invariants that simple np.clip calls do not:
        each threshold stays within its allowed range, AND
        rejection_threshold remains at least 0.05 above acceptance_threshold.
        """

        self.acceptance_threshold = float(np.clip(
            self.acceptance_threshold,
            self.model.min_acceptance_threshold,
            self.model.max_acceptance_threshold,
        ))

        self.rejection_threshold = float(np.clip(
            self.rejection_threshold,
            self.model.min_rejection_threshold,
            self.model.max_rejection_threshold,
        ))

        # Ensure rejection threshold is always above acceptance threshold.
        if self.rejection_threshold <= self.acceptance_threshold:
            self.rejection_threshold = float(np.clip(
                self.acceptance_threshold + 0.05,
                self.model.min_rejection_threshold,
                self.model.max_rejection_threshold,
            ))
