from dataclasses import dataclass
from typing import Dict

@dataclass
class SingleGameSimulationResults:
    """Container for single game simulation results."""

    # Optimal controls
    home_optimal_control: float
    away_optimal_control: float

    # Value functions @ beginning of game
    home_value_t0: float
    away_value_t0: float

    # Expected score differential
    exp_score_diff_q1: float
    exp_score_diff_q2: float
    exp_score_diff_q3: float
    exp_score_diff_q4: float

    # Confidence intervals around score differential
    lower_q1: float
    upper_q1: float
    lower_q2: float
    upper_q2: float
    lower_q3: float
    upper_q3: float
    lower_q4: float
    upper_q4: float

    # Interval probability used
    interval_prob: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "home_optimal_control": self.home_optimal_control,
            "away_optimal_control": self.away_optimal_control,
            "home_value_t0": self.home_value_t0,
            "away_value_t0": self.away_value_t0,
            "exp_score_diff_q1": self.exp_score_diff_q1,
            "exp_score_diff_q2": self.exp_score_diff_q2,
            "exp_score_diff_q3": self.exp_score_diff_q3,
            "exp_score_diff_q4": self.exp_score_diff_q4,
            "lower_q1": self.lower_q1,
            "upper_q1": self.upper_q1,
            "lower_q2": self.lower_q2,
            "upper_q2": self.upper_q2,
            "lower_q3": self.lower_q3,
            "upper_q3": self.upper_q3,
            "lower_q4": self.lower_q4,
            "upper_q4": self.upper_q4,
        }
