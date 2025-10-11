import logging
from typing import Literal, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class SingleGameSim:
    """
    The class enables the simulation, prediction, and evaluation of a single game outcome. 

    Note: While we could take special care to ensure correct scaling, we will most likely
    just use T = 1, as is done in Stern (1994), who models the score differential process
    as a Brownian motion on the time interval [0,1]. 
    """
    def __init__(
        self,
        hbar: float, 
        abar: float,
        pred_nrtg: float,
        dt: Optional[float], 
        N: Optional[int],
        N_sim: int = 100,
        T: float = 1,  
        m_H: float = 1, # Need to estimate with additional linear model
        m_A: float = 1, # Need to estimate with additional linear model
        alpha_H: float = 10, # Need to choose based on risk tolerance
        alpha_A: float = 10, # Need to choose based on risk tolerance
        sigma: float = 15 # Need to estimate with additional linear model
    ) -> None:
        
        if dt is None and N is None:
            raise ValueError("Must specify one of 'dt' and 'N'.")
        if dt is not None and N is not None:
            raise ValueError("Cannot specify both 'dt' and 'N'. Choose one.")
        
        self.T = T # End-of-Game time
        self.hbar = hbar # Home average effort
        self.abar = abar # Away average effort
        self.m_H = m_H # Home scale factor
        self.m_A = m_A # Away scale factor
        self.alpha_H = alpha_H # Home risk factor
        self.alpha_A = alpha_A # Away risk factor
        self.pred_nrtg = pred_nrtg # predictive net rating
        self.X0 = 0 # score differential at beginning of game
        self.sigma = sigma # volatility/diffusion
        self.N_sim = N_sim # number of simulated paths to sample
        
        if N is not None:
            self.N = N
            self.dt = self.T/self.N
        else:
            assert dt is not None
            self.dt = dt
            self.N = int(self.T/self.dt)

        if self.T == 1:
            time_scale = 'Proportion of Game Played'
        else:
            raise ValueError('Unsupported time scale. Please set T equal to 1.')
        
        self.time_vec = np.linspace(0,self.T, self.N+1)

            
        logger.info("="*100)
        logger.info("Time Interval Information")
        logger.info("="*100)
        logger.info(f"Time Scale: {time_scale}")
        logger.info(f"Total Time Interval: [0,{self.T}]")
        logger.info(f"Sub-interval length: Δt = {self.dt}")
        logger.info(f"Number of sub-intervals: {self.N}")
        logger.info(f"Number of time points: {len(self.time_vec)}")
        logger.info("\n")
        logger.info("="*100)
        logger.info("Team Information")
        logger.info("="*100)
        logger.info(f"Home Avg Effort: {self.hbar: .3f}, Away Avg Effort: {self.abar: .3f}")
        logger.info(f"Home Scaling Factor: {self.m_H}, Away Scaling Factor: {self.m_A}")
        logger.info(f"Home Risk: {self.alpha_H}, Away Risk: {self.alpha_A}")
        logger.info(f"Predicted Drift w/o Additional Effort: {self.pred_nrtg: .3f}")

    @staticmethod
    def optimal_control(avg_effort: float, scale_factor: float, risk_factor: float, team: Literal["home","away"]):
        if team == "home":
            return avg_effort + .5 * scale_factor / risk_factor
        elif team == "away":
            return avg_effort - .5 * scale_factor / risk_factor
        else:
            raise ValueError("team must be either 'home' or 'away'")

    
    def value_function(self, t: float, x: float, team: Literal["home", "away"]):
        if team == "home":
            return x + (.25 * self.m_H**2 / self.alpha_H + self.pred_nrtg + self.m_H * self.hbar - self.m_A * self.abar - .5 * self.m_A**2 / self.alpha_A) * (self.T - t)
        elif team == "away":
            return -x + (.25 * self.m_A**2 / self.alpha_A - (self.pred_nrtg + self.m_H * self.hbar - self.m_A * self.abar) - .5 * self.m_H**2 / self.alpha_H) * (self.T - t)
        else: 
            raise ValueError("team must be either 'home' or 'away'")

    @property
    def home_optimal_control(self):
        return self.optimal_control(avg_effort=self.hbar, scale_factor=self.m_H, risk_factor=self.alpha_H, team="home")

    @property
    def away_optimal_control(self):
        return self.optimal_control(avg_effort=self.abar, scale_factor=self.m_A, risk_factor=self.alpha_A, team="away")

    @property
    def home_value_t0(self):
        return self.value_function(t=0, x=0, team="home")

    @property
    def away_value_t0(self):
        return self.value_function(t=0, x=0, team="away")


    def sde_drift(self):
        # Compute optimal controls
        h_opt_control = self.optimal_control(avg_effort=self.hbar, scale_factor=self.m_H, risk_factor=self.alpha_H, team="home")
        a_opt_control = self.optimal_control(avg_effort=self.abar, scale_factor=self.m_A, risk_factor=self.alpha_A, team="away")
        
        return (self.pred_nrtg + self.m_H * h_opt_control - self.m_A * a_opt_control) * self.dt

    def sde_diffusion(self):
        return self.sigma * np.sqrt(self.dt) * norm.rvs(0,1)

    def euler_maruyama_step(self, x0: float):
        """Compute one step in the Euler-Maruyama approximation of the score differential process"""
        drift = self.sde_drift()
        diff = self.sde_diffusion()

        return x0 + drift + diff 

    def euler_maruyama(self):
        """Compute Full Euler-Maruyama Approximation"""
        # Initialize with zeros, keep zeros in first row
        X_mat = np.zeros((self.N+1, self.N_sim))
        
        # Loop over each time after t=0
        for sim in range(self.N_sim):
            for tn in range(1,self.N+1):
                X_mat[tn,sim] = self.euler_maruyama_step(x0=X_mat[tn-1,sim])

        self.X_mat = X_mat

        return self.X_mat

    def plot_paths(self):
        """Plot simulated paths and mean path of score differential process"""
        # Plot each simulated path
        for sim in range(self.N_sim):
            sns.lineplot(x=self.time_vec, y=self.X_mat[:,sim], alpha=.05, color='black')
        
        # Add mean path line
        sns.lineplot(x=self.time_vec, y=self.X_mat.mean(axis=1), color='red', label='E[X_t]')
        plt.legend(loc='best')

    def expected_score_diff(self, t: float):
        return (self.pred_nrtg + self.m_H * self.home_optimal_control - self.m_A * self.away_optimal_control)*t

    def score_diff_interval(self, t: float, interval_prob: float):
        """Compute Score Differential Process Interval Bounds"""
        expected_Xt = self.expected_score_diff(t=t)
        tail_prob = 1-interval_prob
        lower_bound = norm.ppf(q=tail_prob/2, loc=expected_Xt, scale=self.sigma)
        upper_bound = norm.ppf(q=tail_prob/2+interval_prob, loc=expected_Xt, scale=self.sigma)
        return expected_Xt, lower_bound, upper_bound
        

    def results_summary(self, interval_prob: float):
        logger.info("="*50)
        logger.info("Optimal Controls")
        logger.info("="*50)
        logger.info(f"Home Optimal Control: {self.home_optimal_control: .3f}")
        logger.info(f"Away Optimal Control: {self.away_optimal_control: .3f}")

        logger.info("="*50)
        logger.info("Value Functions @ Beginning of Game")
        logger.info("="*50)
        logger.info(f"Home Value @ t = 0: {self.home_value_t0: .3f}")
        logger.info(f"Away Value @ t = 0: {self.away_value_t0: .3f}")

        logger.info("="*50)
        logger.info("Score Differential Predictions")
        logger.info("="*50)
        exp_Xt_q1, lower_q1, upper_q1 = self.score_diff_interval(t=.25, interval_prob=interval_prob)
        exp_Xt_q2, lower_q2, upper_q2 = self.score_diff_interval(t=.50, interval_prob=interval_prob)
        exp_Xt_q3, lower_q3, upper_q3 = self.score_diff_interval(t=.75, interval_prob=interval_prob)
        exp_Xt_q4, lower_q4, upper_q4 = self.score_diff_interval(t=1, interval_prob=interval_prob)
        logger.info(f"Expected Score Diff @ t = 1/4: {exp_Xt_q1: .3f}, w/ {interval_prob*100}% Interval: ({lower_q1: .3f}, {upper_q1: .3f})")
        logger.info(f"Expected Score Diff @ t = 1/2: {exp_Xt_q2: .3f}, w/ {interval_prob*100}% Interval: ({lower_q2: .3f}, {upper_q2: .3f})")
        logger.info(f"Expected Score Diff @ t = 3/4: {exp_Xt_q3: .3f}, w/ {interval_prob*100}% Interval: ({lower_q3: .3f}, {upper_q3: .3f})")
        logger.info(f"Expected Score Diff @ t = 1: {exp_Xt_q4: .3f}, w/ {interval_prob*100}% Interval: ({lower_q4: .3f}, {upper_q4: .3f})")
        
    