import logging
import pandas as pd
from src.steps.model.diff_game_game_theory import SingleGameSim
from src.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

config = Config()

class MultiGameDiffGameGT:
    """
    This class runs the game theoretic formulation of the optimal control problem across multiple games.
    """
    def __init__(self, df_transformed: pd.DataFrame,df_stage2_effort_output: pd.DataFrame) -> None:
        self.df_transformed = df_transformed
        self.df_stage2 = df_stage2_effort_output

        join_cols = ['GAME_ID','AWAY_PTS','HOME_PTS']
        self.df_stage2 = self.df_stage2.merge(
            self.df_transformed[join_cols],
            how='left',
            on='GAME_ID'
        )
    
    def solve_games(self, print_output=False, create_plots=False, save_figs=False):
        # Initialize quantities to collect
        home_optimal_control = []
        away_optimal_control = []
        home_value_t0 = []
        away_value_t0 = []
        expected_Xt = []

        for i, (_, row) in enumerate(self.df_stage2.iterrows()):
            logger.info(f"Game Index: {int(i)+1}")
            game_sim = SingleGameSim(
                hbar=row[f"HOME_AVG{config.feature_creation['window']}"],
                abar=row[f"AWAY_AVG{config.feature_creation['window']}"],
                pred_nrtg=row['pred_nrtg'],
                n_sim=config.simulation['nsim'],
                T=config.simulation['time_scale'],
                dt=None,
                n_points=config.simulation['n_points'],
                m_H=1, # to be estimated with reg model
                m_A=1, # to be estimated with reg model
                alpha_H=10,
                alpha_A=10,
                sigma=15 # to be estimated with reg model
            )

            game_sim.results_summary(interval_prob=config.simulation['interval_prob'])
            home_optimal_control.append(game_sim.home_optimal_control)
            away_optimal_control.append(game_sim.away_optimal_control)
            home_value_t0.append(game_sim.home_value_t0)
            away_value_t0.append(game_sim.away_value_t0)
            expected_Xt.append(game_sim.expected_score_diff(t=game_sim.T))
    
    
