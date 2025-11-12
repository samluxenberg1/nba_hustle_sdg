"""Visualization utilities for model results"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from config.config import config

def plot_effort_parameters(df: pd.DataFrame) -> None:
    """Plot offensive effort, defensive effort, and home advantage over time."""
    plt.figure(figsize=(config['visualization']['figure_size']['width'], 
                        config['visualization']['figure_size']['height']))
    sns.lineplot(data=df, x='game_date', y='off_effort', marker='o', label='Offensive Effort')
    sns.lineplot(data=df, x='game_date', y='def_effort', marker='o', label='Defensive Effort')
    sns.lineplot(data=df, x='game_date', y='home_adv', marker='o', label='Home Advantage')
    plt.title('Effort Effectors Over Time')
    plt.xlabel('Game Date')
    plt.ylabel('Effort Parameter Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_four_factors(df: pd.DataFrame) -> None:
    """Plot the four factors effects over time"""
    plt.figure(figsize=(config['visualization']['figure_size']['width'], 
                        config['visualization']['figure_size']['height']))
    sns.lineplot(data=df, x='game_date', y='avg_efg_pct_diff', marker='o', label='Avg EFG% Diff')
    sns.lineplot(data=df, x='game_date', y='avg_fta_rate_diff', marker='o', label='Avg FTA RATE Diff')
    sns.lineplot(data=df, x='game_date', y='avg_tm_tov_pct_diff', marker='o', label='Avg TOV% Diff')
    sns.lineplot(data=df, x='game_date', y='avg_oreb_pct_diff', marker='o', label='Avg OREB% Diff')
    plt.title('Four Factors Effects Over Time')
    plt.xlabel('Game Date')
    plt.ylabel('Four Factor Parameter Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_net_effort(df: pd.DataFrame) -> None:
    """Plot home and away net effort over time."""
    plt.figure(figsize=(config['visualization']['figure_size']['width'], 
                        config['visualization']['figure_size']['height']))
    sns.lineplot(data=df, x='game_date', y='home_net_effort', marker='o', label='Home Effort')
    sns.lineplot(data=df, x='game_date', y='away_net_effort', marker='o', label='Away Effort')
    plt.title('Home/Away Net Effort Over Time')
    plt.xlabel('Game Date')
    plt.ylabel('Net Effort Parameter Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_drift_parameters(df: pd.DataFrame) -> None:
    """Plot drift parameters m_H and m_A over time."""
    plt.figure(figsize=(config['visualization']['figure_size']['width'], 
                        config['visualization']['figure_size']['height']))
    sns.lineplot(data=df, x='game_date', y='m_H', marker='o', label='m_H')
    sns.lineplot(data=df, x='game_date', y='m_A', marker='o', label='m_A')
    plt.title('Drift Parameters Over Time')
    plt.xlabel('Game Date')
    plt.ylabel('Drift Parameter Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_sigma(df: pd.DataFrame) -> None:
    """Plot sigma parameter over time."""
    plt.figure(figsize=(config['visualization']['figure_size']['width'], 
                        config['visualization']['figure_size']['height']))
    sns.lineplot(data=df, x='game_date', y='sigma', marker='o', color='orange')
    plt.title('Sigma Over Time')
    plt.xlabel('Game Date')
    plt.ylabel('Sigma Value')
    plt.grid()
    plt.tight_layout()
    plt.show()


def create_all_plots(df: pd.DataFrame) -> None:
    """Create all visualization plots."""
    plot_effort_parameters(df)
    plot_four_factors(df)
    plot_net_effort(df)
    plot_drift_parameters(df)
    plot_sigma(df)