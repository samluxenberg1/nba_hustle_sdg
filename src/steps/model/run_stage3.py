import os

from typing import Tuple
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


from src.constants import four_factors_stats
from src.steps.transform.transform_stage3 import TransformStage3
from src.steps.model.model_stage3 import ModelStage3

def run_stage3(
    df_transformed: pd.DataFrame, 
    df_stage1_output: pd.DataFrame, 
    df_train_stage2_output: pd.DataFrame,
    current_date: str, 
    window: int, 
    output_dir: str,
    save_figs: bool = False, 
    print_output: bool = False, 
    create_plots: bool = False,
    save_data: bool = False
) -> Tuple[float, float, float]:
    # Step 1 - Transform Data for Stage 3 Model
    trans_data = TransformStage3(
        df_transformed=df_transformed, 
        df_stage1_output=df_stage1_output, 
        df_train_stage2_output=df_train_stage2_output,
        current_date=current_date
    )

    trans_data.run_transform()
    
    # Prep columns for modeling
    ids = ['SEASON_ID','GAME_ID','GAME_DATE','HOME_TEAM_ID','AWAY_TEAM_ID','HOME_TEAM','AWAY_TEAM']
    target = 'EST_HOME_NRtg' # To be offset by predicted net rating from stage 2 (to constrain coef = 1)
    X_cols = ids + ['HOME_NET_COMPOSITE_EFFORT','AWAY_NET_COMPOSITE_EFFORT']
    offset = 'y_pred_train'

    y = trans_data.df_train_stage2_output[target] - trans_data.df_train_stage2_output[offset]
    X = trans_data.df_train_stage2_output[X_cols]

    model_nrtg = ModelStage3(
        X=X,
        y=y,
        id_cols=ids,
        current_date=current_date,
        model_name='Stage3_Model'
    )

    m_H, m_A, sigma = model_nrtg.run_stage3_model()

    print(f"m_H: {m_H: .2f}, m_A: {m_A: .2f}, sigma: {sigma: .2f}")

    return m_H, m_A, sigma