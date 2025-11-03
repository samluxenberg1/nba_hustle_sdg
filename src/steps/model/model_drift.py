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
df_transformed = pd.read_csv(input_path)
TEST_DATE = pd.to_datetime('2024-10-22')
df_transformed['GAME_DATE_dt'] = pd.to_datetime(df_transformed['GAME_DATE'])
df_transformed.sort_values('GAME_DATE_dt', inplace=True)

dates = df_transformed[df_transformed['GAME_DATE_dt']>=TEST_DATE]['GAME_DATE'].drop_duplicates()

WINDOW = 10
m_H_list = []
m_A_list = []
sigma_list = []

for game_date in dates[:4]:
    df_trans = df_transformed.copy()
    print("="*100)
    num_games = len(df_trans[df_trans['GAME_DATE_dt']==game_date])
    print(f"Game Date: {game_date}, Number of Games: {num_games}")
    print("="*100)
    # Estimate stage 1 effort with all latest data
    df_stage1, model_stage1 = run_stage1(
        df_transformed=df_trans,
        output_dir=stage1_output_dir,
        current_date=game_date, 
        save_figs=False, 
        print_output=False, 
        create_plots=False, 
        save_data=False
    )
    df_stage1['GAME_DATE'] = pd.to_datetime(df_stage1['GAME_DATE'])
    print("Stage 1 effort estimation complete.")
    print("Stage1 Model Summary:")
    print(model_stage1.summary())
    print("Model Coefficient Estimates:")
    print(model_stage1.params)
    

    # Predict current date's net ratings
    df_train_stage2, df_test_stage2, model_stage2 = run_stage2(
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
    print("Stage 2 net rating estimation complete.")
    print("Stage2 Model Summary:")
    print(model_stage2.summary())
    print("Model Coefficient Estimates:")
    print(model_stage2.params)
   


    # Take y_pred_train and use as a predictor of current game net rating for all games prior to today
    # Output: m_H_est, m_A_est, sigma_est
    m_H, m_A, sigma_est, model_stage3 = run_stage3(
        df_transformed=df_trans, 
        df_stage1_output=df_stage1,
        df_train_stage2_output=df_train_stage2,
        current_date=game_date,
        window=WINDOW,
        output_dir=stage3_output_dir
    )

    print("Stage 3 drift parameter estimation complete.")
    print("Stage3 Model Summary:")
    print(model_stage3.summary())
    print("Drift Parameter Estimates:")
    print(f"m_H: {m_H}, m_A: {m_A}, sigma: {sigma_est}")
    m_H_list.append(m_H)
    m_A_list.append(m_A)
    sigma_list.append(sigma_est)

# import seaborn as sns
# import matplotlib.pyplot as plt
# df_drift_params = pd.DataFrame({
#     'GAME_DATE': dates.values,
#     'm_H': m_H_list,
#     'm_A': m_A_list,
#     'sigma': sigma_list
# })
# df_drift_params['GAME_DATE'] = pd.to_datetime(df_drift_params['GAME_DATE'])
# plt.figure(figsize=(12,6))
# sns.lineplot(data=df_drift_params, x='GAME_DATE', y='m_H', marker='o', label='m_H')
# sns.lineplot(data=df_drift_params, x='GAME_DATE', y='m_A', marker='o', label='m_A')
# plt.title('Drift Parameters Over Time')
# plt.xlabel('Game Date')
# plt.ylabel('Drift Parameter Value')
# plt.legend()
# plt.grid()
# plt.show()
# plt.figure(figsize=(12,6))
# sns.lineplot(data=df_drift_params, x='GAME_DATE', y='sigma', marker='o', color='orange')
# plt.title('Sigma Over Time')
# plt.xlabel('Game Date')
# plt.ylabel('Sigma Value')
# plt.grid()
# plt.show()
# print("Drift Parameters DataFrame:")
# print(df_drift_params)