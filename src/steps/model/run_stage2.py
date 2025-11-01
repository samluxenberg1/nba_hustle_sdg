import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


from src.constants import four_factors_stats
from src.steps.transform.transform_stage2 import TransformStage2
from src.steps.model.model_stage2 import ModelStage2

def run_stage2(
    df_transformed: pd.DataFrame, 
    df_stage1_output: pd.DataFrame, 
    current_date: str, 
    window: int, 
    output_dir: str,
    save_figs: bool = False, 
    print_output: bool = False, 
    create_plots: bool = False,
    save_data: bool = False
):
    
    # Transform Data for Stage 2 Model
    trans_data = TransformStage2(
        df_transformed=df_transformed,
        df_stage1_results=df_stage1_output,
        current_date=current_date,
        hist_avg_window=window
    )
    trans_data.run_transform()
    ids = ['SEASON_ID','GAME_ID','GAME_DATE','HOME_TEAM_ID','AWAY_TEAM_ID','HOME_TEAM','AWAY_TEAM', 
           f'HOME_AVG{window}_NET_COMPOSITE_EFFORT',f'AWAY_AVG{window}_NET_COMPOSITE_EFFORT', f'AVG{window}_NET_COMPOSITE_EFFORT_DIFF'] 
    # Prep columns and target
    target = 'EST_HOME_NRtg'
    X_cols = ids + [
        f'AVG{window}_EFG_PCT_DIFF',
        f'AVG{window}_FTA_RATE_DIFF',
        f'AVG{window}_TM_TOV_PCT_DIFF',
        f'AVG{window}_OREB_PCT_DIFF',
        f'AVG{window}_NET_COMPOSITE_EFFORT_DIFF'
    ]

    y_train = trans_data.df_train[target]
    X_train = trans_data.df_train[X_cols]
    
    y_test = trans_data.df_test[target]
    X_test = trans_data.df_test[X_cols]
    

    model_nrtg = ModelStage2(
        X_train=X_train, y_train=y_train, 
        X_test=X_test, 
        model_name="Stage 2 Net Rating", 
        target_name=target, 
        id_cols=ids, 
        current_date=current_date
    )
    
    df_train, df_test = model_nrtg.run_stage2_model(
        output_dir=output_dir,
        save_figs=save_figs,
        print_output=print_output,
        create_plots=create_plots,
        save_data=save_data
    )

    return df_train, df_test
    

if __name__=='__main__':
    import logging
    from src.steps.model.model_utils import daterange
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    DATA_DIR = 'data/'
    transformed_path = os.path.join(DATA_DIR, 'transformed_data', 'df_transformed.csv')
    df_trans = pd.read_csv(transformed_path)
    stage1_path = os.path.join(DATA_DIR, 'stage1_effort', 'df_net_stage1_effort.csv')
    df_stage1 = pd.read_csv(stage1_path)
    output_dir = os.path.join(DATA_DIR, 'stage2_netrating')


    start_date = pd.to_datetime('2024-10-22')
    end_date = pd.to_datetime('2024-10-27')
    
    for date in daterange(start_date=start_date, end_date=end_date):
        
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"\n\n\nRun Stage 2 Model for Game Date: {date_str}")
        df_train, df_test = run_stage2(df_transformed=df_trans, df_stage1_output=df_stage1, output_dir=output_dir,current_date=date_str,window=10)
        logger.info(f"Data Shapes - Train: {df_train.shape}, Test: {df_test.shape}")
        print("="*50)
        print("1st 5 Rows of df_train")
        print("="*50)
        print(df_train.sort_values('GAME_DATE').head())
        print("="*50)
        print("\n")
        print("Last 5 Rows of df_train")
        print("="*50)
        print(df_train.sort_values('GAME_DATE').tail())
        print("="*50)
        print("\n\n")
        print("="*50)
        print("1st 5 Rows of df_test")
        print("="*50)
        print(df_test.sort_values('GAME_DATE').head())
        print("="*50)
        print("\n")
        print("Last 5 Rows of df_test")
        print("="*50)
        print(df_test.sort_values('GAME_DATE').tail())
        print("="*50)

    