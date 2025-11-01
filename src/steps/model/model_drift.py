import os

import pandas as pd

from src.steps.model.run_stage1 import run_stage1
from src.steps.model.run_stage2 import run_stage2
from src.steps.model.run_stage3 import run_stage3


DATA_DIR = 'data/'
input_path = os.path.join(DATA_DIR, 'transformed_data', 'df_transformed.csv')
stage1_output_dir = os.path.join(DATA_DIR, 'stage1_effort')
stage2_output_dir = os.path.join(DATA_DIR, 'stage2_netrating')
stage3_output_dir = os.path.join(DATA_DIR, 'stage3_drift_params_est')

# Start w/ transformed data
df_trans = pd.read_csv(input_path)
TEST_DATE = pd.to_datetime('2024-10-22')
df_trans['GAME_DATE_dt'] = pd.to_datetime(df_trans['GAME_DATE'])
df_trans.sort_values('GAME_DATE_dt', inplace=True)

dates = df_trans[df_trans['GAME_DATE_dt']>=TEST_DATE]['GAME_DATE'].drop_duplicates()

WINDOW = 10

for game_date in dates[:3]:
    print("="*100)
    num_games = len(df_trans[df_trans['GAME_DATE_dt']==game_date])
    print(f"Game Date: {game_date}, Number of Games: {num_games}")
    print("="*100)
    # Estimate stage 1 effort with all latest data
    df_stage1 = run_stage1(
        df_transformed=df_trans,
        output_dir=stage1_output_dir,
        current_date=game_date, 
        save_figs=False, 
        print_output=False, 
        create_plots=False, 
        save_data=False
    )
    df_stage1['GAME_DATE'] = pd.to_datetime(df_stage1['GAME_DATE'])
    

    # Predict current date's net ratings
    df_train_stage2, df_test_stage2 = run_stage2(
        df_transformed=df_trans,
        df_stage1_output=df_stage1,
        output_dir=stage2_output_dir,
        current_date=game_date, 
        window=WINDOW, 
        save_figs=False, 
        print_output=False, 
        create_plots=False, 
        save_data=False
    )
    
    # Join net effort from stage 1 output to df_train
    # df_stage3 = df_train_stage2.merge(
    #     df_stage1[['GAME_ID','TEAM_ID','NET_COMPOSITE_EFFORT']], 
    #     how='left', 
    #     left_on=['GAME_ID','HOME_TEAM_ID'], 
    #     right_on=['GAME_ID','TEAM_ID']
    # )
    # df_stage3.drop('TEAM_ID', axis=1, inplace=True)
    # df_stage3.rename(columns={'NET_COMPOSITE_EFFORT': 'HOME_NET_EFFORT'}, inplace=True)
    # df_stage3 = df_stage3.merge(
    #     df_stage1[['GAME_ID','TEAM_ID','NET_COMPOSITE_EFFORT']],
    #     how='left', 
    #     left_on=['GAME_ID','AWAY_TEAM_ID'],
    #     right_on=['GAME_ID','TEAM_ID']
    # )
    # df_stage3.drop('TEAM_ID',axis=1, inplace=True)
    # df_stage3.rename(columns={'NET_COMPOSITE_EFFORT':'AWAY_NET_EFFORT'}, inplace=True)
    # print("-->"*50)
    # print("First 5 rows of df_stage3")
    # print(df_stage3.sort_values(['GAME_DATE','GAME_ID']).head())
    # print("-->"*50)
    # print("Last 5 rows of df_stage3")
    # print(df_stage3.sort_values(['GAME_DATE','GAME_ID']).tail())
    #print("="*50)
    #print("All Columns")
    #print(df_stage3.columns)


    # Take y_pred_train and use as a predictor of current game net rating for all games prior to today
    # Output: m_H_est, m_A_est, sigma_est
    m_H, m_A, sigma_est = run_stage3(
        df_transformed=df_trans, 
        df_stage1_output=df_stage1,
        df_train_stage2_output=df_train_stage2,
        current_date=game_date,
        window=WINDOW,
        output_dir=stage3_output_dir
    )