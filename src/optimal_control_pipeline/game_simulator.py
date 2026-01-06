"""Wrapper for single game simulation and optimal control"""
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple

from src.simulation.single_game_sim import SingleGameSim
from config.config import config

@dataclass
class GameParameters:
    """Parameters for a single game simulation"""
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    hbar: float
    abar: float
    pred_nrtg: float
    m_h: float
    m_a: float
    sigma: float
    home_poss_factor: float
    away_poss_factor: float
    alpha_H: float = 0.1
    alpha_A: float = 0.1

    @property
    def avg_poss_factor(self) -> float:
        """Calculate average possession factor"""
        return (self.home_poss_factor + self.away_poss_factor) / 2
    
    def get_adjusted_parameters(self) -> Dict[str, float]:
        """
        Adjust parameters by possession factor

        Returns:
            Dictionary of adjusted parameters
        """
        scale = self.avg_poss_factor / 100
        return {
            'm_h_adj': self.m_h * scale,
            'm_a_adj': self.m_a * scale,
            'sigma_adj': self.sigma * scale,
            'pred_nrtg_adj': self.pred_nrtg * scale
        }
    
    def print_summary(self) -> None:
        """Print parameter summary"""
        adj = self.get_adjusted_parameters()
        print(f"home_poss_factor: {self.home_poss_factor: .3f},"
              f"away_poss_factor: {self.away_poss_factor: .3f},"
              f"avg_poss_factor: {self.avg_poss_factor: .3f}")
        print(f"hbar: {self.hbar: .3f}, abar: {self.abar: .3f}")
        print(f"m_h: {self.m_h: .3f} -> adj_m_h: {adj['m_h_adj']: .3f}, "
              f"m_a: {self.m_a: .3f} -> adj_m_a: {adj['m_a_adj']: .3f}")
        print(f"pred_nrtg: {self.pred_nrtg: .3f} -> adj_pred_nrtg: {adj['pred_nrtg_adj']: .3f}")
        print(f"sigma: {self.sigma: .3f} -> adj_sigma: {adj['sigma_adj']: .3f}")

class GameSimulator: 
    """
    Wrapper for running single game simulations

    This class handles the optimal control problem for individual games using the SingleGameSim class.
    """
    def __init__(self) -> None:
        """
        Initialize the game simulator.

        Args:
            config: Configuration dictionary with simulation parameters
        """
        self.n_points = config['simulation']['n_points']
        self.n_sim = config['simulation']['n_sim']
        self.T = config['simulation']['T']
        self.interval_prob = config['simulation']['interval_prob']

    def run_simulation(self, params: GameParameters, show_plots: bool = True) -> SingleGameSim:
        """
        Run optimal control simulation for a single game. 

        Args:
            params: Game parameters
            show_plots: Whether to display plots

        Returns:
            SingleGameSim instance with results
        """
        adj = params.get_adjusted_parameters()

        sgs = SingleGameSim(
            hbar=params.hbar,
            abar=params.abar,
            pred_nrtg=adj['pred_nrtg_adj'],
            dt=None,
            n_points=self.n_points,
            n_sim=self.n_sim,
            T=self.T,
            m_H=adj['m_h_adj'],
            m_A=adj['m_a_adj'],
            alpha_H=params.alpha_H,
            alpha_A=params.alpha_A,
            sigma=adj['sigma_adj'],
            interval_prob=self.interval_prob
        )

        print("="*100)
        print(f"Differential Game Results: {params.home_team} vs {params.away_team} on {params.game_date}; Game ID: {params.game_id}")
        print("="*100)

        sgs.results_summary()
        sgs.euler_maruyama()

        if show_plots:
            sgs.plot_paths()
            plt.show()

        return sgs
    
    def simulate_all_games(
            self, 
            df_test: pd.DataFrame, 
            df_poss: pd.DataFrame, 
            m_H: float, 
            m_A: float, 
            sigma: float, 
            window: int = config['possession']['window'], 
            show_plots: bool = True
        ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run simulations for all games in test set.

        Args:
            df_test: Test data with predictions
            df_poss: DataFrame with possession factors
            m_H: Home scale parameter
            m_A: Away scale parameter
            sigma: Volatility parameter
            window: Window size for effort trends
            show_plots: Whether to display plots

        Returns:
            Tuple of (simulation objects dict, results DataFrame)
        """
        print("$"*150)
        print("Optimal Control Parameters")
        print("$"*150)
        print(f"df_test columns: {df_test.columns.tolist()}")
        print(df_test.head())

        results = {}
        results_list = []

        for index, row in df_test.iterrows():
            # Extract game parameters
            home_team = row['HOME_TEAM']
            away_team = row['AWAY_TEAM']

            params = GameParameters(
                game_id=row['GAME_ID'],
                game_date=row['GAME_DATE'],
                home_team=home_team, 
                away_team=away_team,
                hbar=row[f'HOME_AVG{window}_NET_COMPOSITE_EFFORT'],
                abar=row[f'AWAY_AVG{window}_NET_COMPOSITE_EFFORT'],
                pred_nrtg=row['y_pred_test'],
                m_h=m_H,
                m_a=m_A,
                sigma=sigma,
                home_poss_factor=df_poss[df_poss.index==home_team]['home_avg_poss'].values[0],
                away_poss_factor=df_poss[df_poss.index==away_team]['away_avg_poss'].values[0]
            )

            # Print parameter summary
            params.print_summary()

            # Run simulation
            sgs = self.run_simulation(params, show_plots)
            sgs.compute_and_store_results()

            # Store simulation object
            results[(home_team, away_team)] = sgs

            # Get results as dict and add game info
            game_results = sgs.to_dataframe().iloc[0].to_dict()
            game_results['home_team'] = home_team
            game_results['away_team'] = away_team
            game_results['game_date'] = row.get('GAME_DATE', None)

            results_list.append(game_results)

            print("\n\n")

        # Create DataFrame from all results
        results_df = pd.DataFrame(results_list)
        
        return results, results_df


