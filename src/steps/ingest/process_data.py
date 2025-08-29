import os
from typing import Dict
import pandas as pd
import numpy as np

class ProcessData:
    def __init__(self, df_logs: pd.DataFrame, df_hustle: pd.DataFrame, neutral_dict: Dict):
        self.df_logs = df_logs
        self.df_hustle = df_hustle
        self.neutral_dict = neutral_dict
    
    def process_logs(self) -> pd.DataFrame:
        
        df = self.df_logs.copy()
        
        # Add home team indicator
        df['HOME_IND'] = np.where(df['MATCHUP'].str.contains('@'),'AWAY','HOME')

        # Handle handful of international games manually --> figure out more robust solution later...
        neutral_games_idx = []
        for game_id, home_team in self.neutral_dict.items():
            neutral_games_idx.append(df[(df['GAME_ID']==game_id) & (df['TEAM_ABBREVIATION']==home_team)].index.values[0])
        df.loc[neutral_games_idx, 'HOME_IND'] = 'HOME'
        
        
        # Define columns
        team_cols = ['TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PF','PTS']
        non_team_cols = ['SEASON_ID','GAME_ID','GAME_DATE']

        # Pivot data (convert 2 rows per game to 1 row per game)
        df_pivoted = df.set_index(non_team_cols+['HOME_IND'])[team_cols].unstack('HOME_IND')

        # Flatten column names
        df_pivoted.columns = [f"{home_ind}_{col}" for col, home_ind in df_pivoted.columns]
        self.df_proc_logs = df_pivoted.reset_index()
        
        # Add neutral indicator column
        self.df_proc_logs['NEUTRAL_IND'] = 0
        self.df_proc_logs.loc[self.df_proc_logs['GAME_ID'].isin(list(self.neutral_dict.keys())), 'NEUTRAL_IND'] = 1

        return self.df_proc_logs

    def process_hustle(self):
        df_hustle = self.df_hustle.copy()
        df_logs = self.df_proc_logs.copy()

        # Join home indicator to hustle data (already accounts for neutral games from df_proc_logs)
        df_home_ind = df_logs[['GAME_ID','HOME_TEAM_ID']].copy()
        df_home_ind['HOME_IND'] = 'HOME'
        df_hustle = df_hustle.merge(
            df_home_ind, 
            left_on=['GAME_ID','TEAM_ID'],
            right_on=['GAME_ID','HOME_TEAM_ID'],
            how='left'
        )
        df_hustle.loc[df_hustle['HOME_IND'].isna(), 'HOME_IND'] = 'AWAY'
        
        # Drop unnecessary columns
        cols_to_drop = ['TEAM_NAME','TEAM_ABBREVIATION','TEAM_CITY','MINUTES','PTS','HOME_TEAM_ID']
        df_hustle.drop(cols_to_drop, axis=1, inplace=True)

        # Convert from long to wide
        team_cols = list(set(df_hustle.columns)-set(['GAME_ID','HOME_IND']))
        df_hustle_pivoted = df_hustle.set_index(['GAME_ID','HOME_IND'])[team_cols].unstack('HOME_IND')

        df_hustle_pivoted.columns = [f"{home_ind}_{col}" for col, home_ind in df_hustle_pivoted.columns]
        self.df_proc_hustle = df_hustle_pivoted.reset_index()

        return self.df_proc_hustle
    
    def process_data(self):
        
        # Step 1 - Process Game Logs
        df_proc_logs = self.process_logs()

        # Step 2 - Process Hustle Data
        df_proc_hustle = self.process_hustle()

        # Step 3 - Join Game Logs and Hustle
        df_proc_hustle.drop(['HOME_TEAM_ID','AWAY_TEAM_ID'], axis=1, inplace=True)
        self.df_proc = self.df_proc_logs.merge(
            self.df_proc_hustle, 
            on='GAME_ID',
            how='inner'
        )

        return self.df_proc






if __name__=='__main__':
    # Read in log data
    DATA_DIR = '../../data/'
    lgl_path = os.path.join(DATA_DIR, 'df_logs.csv')
    df_logs = pd.read_csv(lgl_path)
    hustle_files = [f for f in os.listdir(DATA_DIR) if f.startswith('df_hustle')]
    hustle_paths = [os.path.join(DATA_DIR, f) for f in hustle_files]
    df_hustle_list = [pd.read_csv(f) for f in hustle_paths]
    df_hustle = pd.concat(df_hustle_list)
    
    neutral_game_ids = [22400147, 22401230, 22401229, 22400621, 22400633]
    neutral_home_teams = ['WAS','OKC','ATL','IND','SAS']
    neutral_dict = dict(zip(neutral_game_ids,neutral_home_teams))
    
    proc_data = ProcessData(df_logs = df_logs, df_hustle=df_hustle, neutral_dict=neutral_dict)
    df_proc = proc_data.process_data()

    print(df_proc.head())