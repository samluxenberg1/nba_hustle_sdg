"""Utilities for collecting and organizing model parameters"""
from dataclasses import dataclass, field
from typing import List
import pandas as pd

@dataclass
class ModelParameters:
    """Container for all model parameters across stages."""

    # Stage 1 parameters
    off_effort: List[float] = field(default_factory=list)
    def_effort: List[float] = field(default_factory=list)

    # Stage 2 parameters
    home_adv: List[float] = field(default_factory=list)
    avg_efg_pct_diff: List[float] = field(default_factory=list)
    avg_fta_rate_diff: List[float] = field(default_factory=list)
    avg_tm_tov_pct_diff: List[float] = field(default_factory=list)
    avg_oreb_pct_diff: List[float] = field(default_factory=list)

    # Stage 3 parameters
    home_net_effort: List[float] = field(default_factory=list)
    away_net_effort: List[float] = field(default_factory=list)
    m_H: List[float] = field(default_factory=list)
    m_A: List[float] = field(default_factory=list)
    sigma: List[float] = field(default_factory=list)

    # Dates
    dates: List[str] = field(default_factory=list)

    def add_stage1_params(self, model) -> None:
        """Extract and store Stage 1 parameters."""
        self.off_effort.append(model.params.iloc[1])
        self.def_effort.append(model.params.iloc[2])

    def add_stage2_params(self, model) -> None:
        """Extract and store Stage 2 parameters."""
        self.home_adv.append(model.params.iloc[0])
        self.avg_efg_pct_diff.append(model.params.iloc[1])
        self.avg_fta_rate_diff.append(model.params.iloc[2])
        self.avg_tm_tov_pct_diff.append(model.params.iloc[3])
        self.avg_oreb_pct_diff.append(model.params.iloc[4])
    
    def add_stage3_params(self, m_H: float, m_A: float, sigma: float, model) -> None:
        """Extract and store Stage 3 parameters."""
        self.home_net_effort.append(model.params.iloc[0])
        self.away_net_effort.append(model.params.iloc[1])
        self.m_H.append(m_H)
        self.m_A.append(m_A)
        self.sigma.append(sigma)

    def add_date(self, date: str) -> None:
        """Add a date to the collection."""
        self.dates.append(date)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected parameters to a DataFrame."""
        return pd.DataFrame({
            'game_date': pd.to_datetime(self.dates),
            'off_effort': self.off_effort,
            'def_effort': self.def_effort,
            'home_adv': self.home_adv,
            'avg_efg_pct_diff': self.avg_efg_pct_diff,
            'avg_fta_rate_diff': self.avg_fta_rate_diff,
            'avg_tm_tov_pct_diff': self.avg_tm_tov_pct_diff,
            'avg_oreb_pct_diff': self.avg_oreb_pct_diff,
            'home_net_effort': self.home_net_effort,
            'away_net_effort': self.away_net_effort,
            'm_H': self.m_H,
            'm_A': self.m_A,
            'sigma': self.sigma
        })