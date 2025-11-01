import os
from typing import List, Literal
import pandas as pd
import numpy as np

from src.steps.model.model_stage1 import ModelStage1
from src.constants import hustle_stats, home_away_id_cols
from src.config import Config

import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def prep_off_def_model_data(
        df_train: pd.DataFrame, 
        features: List[str], 
        target_name: str, 
        effort_type: Literal['off','def','net'],
        model_name: str
    ) -> ModelStage1:
    """Prepare data for input into offensive, defensive rating models"""

    home_features = [f"HOME_{stat}" for stat in features] 
    away_features = [f"AWAY_{stat}" for stat in features]
    home_ids = ['SEASON_ID','GAME_ID','GAME_DATE','HOME_TEAM_ID','HOME_TEAM_ABBREVIATION'] 
    away_ids = ['SEASON_ID','GAME_ID','GAME_DATE','AWAY_TEAM_ID','AWAY_TEAM_ABBREVIATION']
    home_features_plus_ids = home_ids + home_features
    away_features_plus_ids = away_ids + away_features
    X_home = df_train[home_features_plus_ids]
    X_away = df_train[away_features_plus_ids]

    # Rename columns
    X_home.columns = X_home.columns.str.replace('HOME_','')
    X_away.columns = X_away.columns.str.replace('AWAY_','')
    ids = [id.replace('HOME_','') for id in home_ids]

    # Define target
    y_home = df_train[f'EST_HOME_{target_name}'] 
    y_away = df_train[f'EST_AWAY_{target_name}'] 

    # Stack - Home | Away
    X = pd.concat([X_home, X_away], ignore_index=True)
    y = pd.concat([y_home, y_away], ignore_index=True)
    
    return ModelStage1(
        X=X, 
        y=y, 
        effort_type=effort_type, 
        id_cols=ids, 
        model_name=model_name, 
        target_name=target_name
    )

def prep_net_model_data(
        df_train: pd.DataFrame, 
        off_effort: pd.DataFrame, 
        def_effort: pd.DataFrame,
        model_name: str
    ) -> ModelStage1:
    """Prepare data for input into net rating model"""
    # Define target
    y_off_home = df_train['EST_HOME_ORtg']
    y_def_home = df_train['EST_HOME_DRtg'] # HOME_DRtg = AWAY_ORtg
    y_net_home = y_off_home - y_def_home

    y_off_away = df_train['EST_AWAY_ORtg']
    y_def_away = df_train['EST_AWAY_DRtg'] # AWAY_DRtg = HOME_ORtg
    y_net_away = y_off_away - y_def_away

    y_net = pd.concat([y_net_home, y_net_away], ignore_index=True)

    # Define X
    ids = ['SEASON_ID','GAME_ID','GAME_DATE','TEAM_ID','TEAM_ABBREVIATION'] 
    X_net = off_effort.merge(
        def_effort.drop(['SEASON_ID','GAME_DATE','TEAM_ABBREVIATION'],axis=1), 
        how='inner', 
        on=['GAME_ID','TEAM_ID']
    )
    X_cols = ids + ['OFF_COMPOSITE_EFFORT','DEF_COMPOSITE_EFFORT']

    return ModelStage1(
        X=X_net[X_cols], 
        y=y_net, 
        effort_type='net', 
        id_cols=ids,
        model_name=model_name,
        target_name='EST_NRtg'
    )


def run_stage1(
        df_transformed: pd.DataFrame,
        current_date: str,
        output_dir: str, 
        save_figs: bool = False, 
        print_output: bool = False, 
        create_plots: bool = False,
        save_data: bool = False
    ) -> pd.DataFrame:
    # Read in data
    logger.info("Read in data...")
    logger.info(f"df_trans: {df_transformed.shape}")

    config = Config()

    # Convert game date to datetime to split
    logger.info("Convert GAME_DATE to datetime...")
    df_transformed['GAME_DATE'] = pd.to_datetime(df_transformed['GAME_DATE'])

    # Split
    logger.info(f"Split transformed data on {current_date}")
    #df_trans_score = df_trans[df_trans['GAME_DATE']==current_date]
    df_trans_train = df_transformed[df_transformed['GAME_DATE']<current_date]
    logger.info(f"df_trans_train: {df_trans_train.shape}")

    # Offensive Rating Model
    logger.info("Estimating offensive effort...")
    comp_eff_off = prep_off_def_model_data(
        df_train=df_trans_train, 
        features=config.features_stage1['offensive_features'],
        target_name='ORtg',
        effort_type='off',
        model_name='Offensive Rating'
    )
    off_stage1 = comp_eff_off.run_stage1_model(
        output_dir=output_dir,
        save_figs=save_figs,
        print_output=print_output,
        create_plots=create_plots,
        save_data=save_data
    )
    
    logger.info("Offensive effort complete.")

    # Defensive Rating Model
    logger.info("Estimating defensive effort...")
    comp_eff_def = prep_off_def_model_data(
        df_train=df_trans_train, 
        features=config.features_stage1['defensive_features'], 
        target_name='DRtg',
        effort_type='def',
        model_name='Defensive Rating'
    )
    def_stage1 = comp_eff_def.run_stage1_model(
        output_dir=output_dir,
        save_figs=save_figs,
        print_output=print_output,
        create_plots=create_plots,
        save_data=save_data
    )
    
    logger.info("Defensive effort complete.")

    # Net Rating Model
    logger.info("Estimating net effort...")
    comp_eff_net = prep_net_model_data(
        df_train=df_trans_train,
        off_effort=comp_eff_off.X_with_ids,
        def_effort=comp_eff_def.X_with_ids,
        model_name='Net Rating'
    )
    net_stage1 = comp_eff_net.run_stage1_model(
        output_dir=output_dir,
        save_figs=save_figs,
        print_output=print_output,
        create_plots=create_plots,
        save_data=save_data
    )
    
    logger.info("Net effort complete.")

    return net_stage1

if __name__=='__main__':

    DATA_DIR = 'data/'
    input_path = os.path.join(DATA_DIR, 'transformed_data', 'df_transformed.csv')
    output_dir = os.path.join(DATA_DIR, 'stage1_effort')
    df_trans = pd.read_csv(input_path)

    current_date_list = ['2023-10-22', '2024-01-01','2024-10-22']
    current_date_list = ['2024-10-22']
    for date in current_date_list:
        run_stage1(df_transformed=df_trans, output_dir=output_dir, current_date=date, print_output=True)
