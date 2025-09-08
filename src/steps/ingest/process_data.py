import os
from typing import Dict
import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ProcessData:
    def __init__(self, df_logs: pd.DataFrame, df_hustle: pd.DataFrame, df_four_factors: pd.DataFrame, neutral_dict: Dict):
        self.df_logs = df_logs
        self.df_hustle = df_hustle
        self.df_ff = df_four_factors
        self.neutral_dict = neutral_dict

        # Drop duplicate rows
        self.df_logs = self.df_logs.drop_duplicates()
        self.df_hustle = self.df_hustle.drop_duplicates()
        self.df_ff = self.df_ff.drop_duplicates()

        logger.info(f"df_logs shape: {self.df_logs.shape}")
        logger.info(f"df_hustle shape: {self.df_hustle.shape}")
        logger.info(f"df_four_factors shape: {self.df_ff.shape}")
    
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
        logger.info("Converting long to wide format...")
        df_pivoted = df.set_index(non_team_cols+['HOME_IND'])[team_cols].unstack('HOME_IND')

        # Flatten column names
        df_pivoted.columns = [f"{home_ind}_{col}" for col, home_ind in df_pivoted.columns]
        self.df_proc_logs = df_pivoted.reset_index()
        
        # Add neutral indicator column
        self.df_proc_logs['NEUTRAL_IND'] = 0
        self.df_proc_logs.loc[self.df_proc_logs['GAME_ID'].isin(list(self.neutral_dict.keys())), 'NEUTRAL_IND'] = 1
        logger.info("Added neutral and home indicator columns")

        return self.df_proc_logs

    @staticmethod
    def process_boxscore_data(df_box: pd.DataFrame, df_proc_logs: pd.DataFrame, cols_to_drop) -> pd.DataFrame:

        # Join home indicator to box score data
        df_home_ind = df_proc_logs[['GAME_ID','HOME_TEAM_ID']].copy()
        df_home_ind['HOME_IND'] = 'HOME'
        df_box = df_box.merge(
            df_home_ind,
            left_on=['GAME_ID','TEAM_ID'],
            right_on=['GAME_ID','HOME_TEAM_ID'],
            how='left'
        )
        df_box.loc[df_box['HOME_IND'].isna(), 'HOME_IND'] = 'AWAY'

        # Drop unnecessary columns
        df_box.drop(cols_to_drop, axis=1, inplace=True)

        # Convert from long to wide
        team_cols = list(set(df_box.columns)-set(['GAME_ID','HOME_IND']))
        df_box_pivoted = df_box.set_index(['GAME_ID','HOME_IND'])[team_cols].unstack('HOME_IND')
        df_box_pivoted.columns = [f"{home_ind}_{col}" for col, home_ind in df_box_pivoted.columns]
        
        return df_box_pivoted.reset_index()
    
    
    def process_data(self, output_dir: str):
        
        # Step 1 - Process Game Logs
        logger.info("Process df_logs...")
        df_proc_logs = self.process_logs()
        logger.info("Processing df_logs complete")
        logger.info(f"df_proc_logs shape: {df_proc_logs.shape}")

        # Step 2 - Process Hustle Data
        logger.info("Process df_hustle data...")
        df_proc_hustle = self.process_boxscore_data(
            df_box=self.df_hustle, 
            df_proc_logs=df_proc_logs,
            cols_to_drop=['TEAM_NAME','TEAM_ABBREVIATION','TEAM_CITY','MINUTES','PTS','HOME_TEAM_ID']
        )
        logger.info("Processing df_hustle complete")
        logger.info(f"df_proc_hustle shape: {df_proc_hustle.shape}")

        # Step 3 - Process Four Factors Data
        logger.info("Process df_four_factors...")
        df_proc_ff = self.process_boxscore_data(
            df_box=self.df_ff,
            df_proc_logs=df_proc_logs,
            cols_to_drop=['TEAM_NAME','TEAM_ABBREVIATION','TEAM_CITY','MIN','HOME_TEAM_ID','OPP_EFG_PCT','OPP_FTA_RATE','OPP_TOV_PCT','OPP_OREB_PCT']
        )
        logger.info("Processing df_four_factors complete")
        logger.info(f"df_proc_four_factors shape: {df_proc_ff.shape}")

        # Step 4 - Join Game Logs, Hustle, Four Factors
        logger.info("Joining df_proc_hustle and df_proc_four_factors to df_logs...")
        df_proc_hustle.drop(['HOME_TEAM_ID','AWAY_TEAM_ID'], axis=1, inplace=True)
        df_proc_ff.drop(['HOME_TEAM_ID','AWAY_TEAM_ID'], axis=1, inplace=True)
        self.df_proc = df_proc_logs.merge(
            df_proc_hustle, 
            on='GAME_ID',
            how='inner'
        ).merge(
            df_proc_ff,
            on='GAME_ID',
            how='inner'
        )
        logger.info("Join complete")
        logger.info(f"df_proc shape: {self.df_proc.shape}")

        # Step 4 - Save data
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'df_processed.csv')
        self.df_proc.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

        return self.df_proc






if __name__=='__main__':
    # Read in log data
    

    DATA_DIR = 'data/raw_data/'
    lgl_path = os.path.join(DATA_DIR, 'df_logs.csv')

    df_logs = pd.read_csv(lgl_path)
    hustle_files = [f for f in os.listdir(DATA_DIR) if f.startswith('df_hustle')]
    hustle_paths = [os.path.join(DATA_DIR, f) for f in hustle_files]
    df_hustle_list = [pd.read_csv(f) for f in hustle_paths]
    df_hustle = pd.concat(df_hustle_list)
    df_hustle = df_hustle.drop_duplicates()
    ff_files = [f for f in os.listdir(DATA_DIR) if f.startswith('df_four_factors')]
    ff_paths = [os.path.join(DATA_DIR, f) for f in ff_files]
    df_ff_list = [pd.read_csv(f) for f in ff_paths]
    df_ff = pd.concat(df_ff_list)
    df_ff.drop_duplicates()

    
    
    neutral_game_ids = [22400147, 22401230, 22401229, 22400621, 22400633]
    neutral_home_teams = ['WAS','OKC','ATL','IND','SAS']
    neutral_dict = dict(zip(neutral_game_ids,neutral_home_teams))
    
    proc_data = ProcessData(df_logs = df_logs, df_hustle=df_hustle, df_four_factors=df_ff, neutral_dict=neutral_dict)
    
    output_dir = os.path.join(DATA_DIR, 'processed_data')
    df_proc = proc_data.process_data(output_dir=output_dir)

    print(df_proc.head())