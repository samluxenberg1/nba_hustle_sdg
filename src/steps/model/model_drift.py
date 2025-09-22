import os

import pandas as pd

from src.steps.model.run_stage1 import run_stage1
from src.steps.model.run_stage2 import run_stage2


DATA_DIR = 'data/'
input_path = os.path.join(DATA_DIR, 'transformed_data', 'df_transformed.csv')

# Start w/ transformed data
df_trans = pd.read_csv(input_path)
TEST_DATE = pd.to_datetime('2024-10-22')
df_trans['GAME_DATE_dt'] = pd.to_datetime(df_trans['GAME_DATE'])
df_trans.sort_values('GAME_DATE_dt', inplace=True)

dates = df_trans[df_trans['GAME_DATE_dt']>=TEST_DATE]['GAME_DATE'].drop_duplicates()

WINDOW = 10

for game_date in dates[:3]:

    print(f"Date: {game_date}")
    # Estimate stage 1 effort with all latest data
    run_stage1(current_date=game_date, save_figs=False, print_output=False, create_plots=False)

    # Predict current date's net ratings
    run_stage2(current_date=game_date, window=WINDOW, save_figs=False, print_output=False, create_plots=False)