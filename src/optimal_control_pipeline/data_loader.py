"""Data loading and preprocessing utilities"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

def load_and_prepare_data(
        input_path: Path, 
        test_start_date: pd.Timestamp, 
        test_end_date: Optional[pd.Timestamp] = None
        ) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from CSV and prepare training and testing datasets.

    Args:
        input_path (Path): Path to the input CSV file.
        test_date (pd.Timestamp): Date to split training and testing data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
    """
    # Load data
    df = pd.read_csv(input_path, parse_dates=['GAME_DATE'])
    
    # Ensure date column is in datetime format
    df['GAME_DATE_dt'] = pd.to_datetime(df['GAME_DATE'])
    df.sort_values('GAME_DATE_dt', inplace=True)

    dates_mask = (df['GAME_DATE_dt'] >= test_start_date) & (df['GAME_DATE_dt'] <= test_end_date)
    dates = df[dates_mask]['GAME_DATE_dt'].drop_duplicates()

    return df, dates

def print_data_summary(df: pd.DataFrame, game_date: str) -> None:
    """Print summary statistics for a given game date.

    Args:
        df (pd.DataFrame): DataFrame with game data
        game_date (str): Date to summarize
    """
    game_date_dt = pd.to_datetime(game_date)
    num_games = len(df[df['GAME_DATE_dt'] == game_date_dt])

    print("=" * 100)
    print(f"Game Date: {game_date}, Number of Games: {num_games}")
    print("=" * 100)
    print("Transformed Data Sample:")
    cols = ['GAME_ID','GAME_DATE','HOME_TEAM_ABBREVIATION','AWAY_TEAM_ABBREVIATION','HOME_PTS','AWAY_PTS']
    print(df[df['GAME_DATE_dt'] == game_date_dt][cols].head())

    
    