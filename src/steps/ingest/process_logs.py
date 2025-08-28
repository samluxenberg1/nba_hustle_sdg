import os

import pandas as pd
import numpy as np

def process_logs(df: pd.DataFrame, neutral_dict: dict) -> pd.DataFrame:
    # Add home team indicator
    df['HOME_IND'] = np.where(df['MATCHUP'].str.contains('@'),'AWAY','HOME')

    # Handle handful of international games manually --> figure out more robust solution later...
    neutral_games_idx = []
    for game_id, home_team in neutral_dict.items():
        neutral_games_idx.append(df[(df['GAME_ID']==game_id) & (df['TEAM_ABBREVIATION']==home_team)].index.values[0])
    df.loc[neutral_games_idx, 'HOME_IND'] = 'HOME'
    
    
    # Define columns
    team_cols = ['TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PF','PTS']
    non_team_cols = ['SEASON_ID','GAME_ID','GAME_DATE']

    # Pivot data (convert 2 rows per game to 1 row per game)
    df_pivoted = df.set_index(non_team_cols+['HOME_IND'])[team_cols].unstack('HOME_IND')

    # Flatten column names
    df_pivoted.columns = [f"{home_ind}_{col}" for col, home_ind in df_pivoted.columns]
    df_result = df_pivoted.reset_index()
    
    # Add neutral indicator column
    df_result['NEUTRAL_IND'] = 0
    df_result.loc[df_result['GAME_ID'].isin(list(neutral_dict.keys())), 'NEUTRAL_IND'] = 1

    return df_result


if __name__=='__main__':
    # Read in log data
    DATA_DIR = '../../data/'
    lgl_path = os.path.join(DATA_DIR, 'df_logs.csv')
    df_logs = pd.read_csv(lgl_path)
    
    neutral_game_ids = [22400147, 22401230, 22401229, 22400621, 22400633]
    neutral_home_teams = ['WAS','OKC','ATL','IND','SAS']
    neutral_dict = dict(zip(neutral_game_ids,neutral_home_teams))
    df_proc = process_logs(df=df_logs, neutral_dict=neutral_dict)

    print(df_proc.head())