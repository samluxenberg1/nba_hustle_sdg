"""Utilities for calculating possession factors."""
import pandas as pd
from typing import Tuple, Optional

class PossessionCalculator:
    """Calculate team possession factors for parameter adjustments."""

    @staticmethod
    def calculate_possession_factors(
        df: pd.DataFrame,
        game_date: pd.Timestamp,
        window: int,
        group_col: str = 'TEAM_ABBREVIATION',
        possession_col: str = 'EST_POSS'
    ) -> pd.DataFrame:
        """
        Calculate rolling average possession factors for each team.

        Args:
            df: Transformed data with possession estimates
            game_date: Current game date
            window: Number of games to include in rolling average

        Returns:
            DataFrame with home_avg_poss and away_avg_poss by team
        """
        # Get data prior to current date
        df_prior = df[df['GAME_DATE_dt'] < game_date].copy()

        # Grouping columns
        home_group_col = f'HOME_{group_col}'
        away_group_col = f'AWAY_{group_col}'

        # Possessions columns
        home_poss_col = f'HOME_{possession_col}'
        away_poss_col = f'AWAY_{possession_col}'

        # Calculate rolling home possession average
        h_avg_poss = (
            df_prior
            .groupby(home_group_col)
            .tail(window)
            .groupby(home_group_col)[home_poss_col]
            .mean()
            )
        
        a_avg_poss = (
            df_prior
            .groupby(away_group_col)
            .tail(window)
            .groupby(away_group_col)[away_poss_col]
            .mean()
        )
        df_avg_poss = pd.DataFrame(
            {
                'home_avg_poss': h_avg_poss,
                'away_avg_poss': a_avg_poss
            }
        )

        return df_avg_poss
    
    @staticmethod
    def print_possession_sumary(df_poss: pd.DataFrame, team: Optional[str]) -> None:
        """
        Print possession factor sumary

        Args:
            df_poss: DataFrame with possession factors
            team: Specific team to summarize
        """
        if team:
            print(f"Possession Factors for {team}:")
            print(df_poss[df_poss.index==team])
            
        else:
            print("Possession Factors:")
            print(df_poss)
        print("%"*50)