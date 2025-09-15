import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import four_factors_stats
from src.steps.transform.transform_stage2 import TransformStage2

# def team_schedule(df: pd.DataFrame) -> pd.DataFrame:
#     cols = [
#         'GAME_ID','HOME_TEAM','AWAY_TEAM','GAME_DATE',
#         'HOME_EFG_PCT','AWAY_EFG_PCT', 
#         'HOME_TM_TOV_PCT','AWAY_TM_TOV_PCT',
#         'HOME_FTA_RATE','AWAY_FTA_RATE',
#         'HOME_OREB_PCT','AWAY_OREB_PCT',
#         'HOME_NET_COMPOSITE_EFFORT','AWAY_NET_COMPOSITE_EFFORT'
#     ]
#     df_home = (
#         df[cols]
#         .rename(columns={
#             'HOME_TEAM':'TEAM',
#             'AWAY_TEAM':'OPP_TEAM',
#             'HOME_EFG_PCT':'TEAM_EFG_PCT',
#             'AWAY_EFG_PCT':'OPP_EFG_PCT',
#             'HOME_TM_TOV_PCT': 'TEAM_TM_TOV_PCT',
#             'AWAY_TM_TOV_PCT':'OPP_TOV_PCT',
#             'HOME_FTA_RATE':'TEAM_FTA_RATE',
#             'AWAY_FTA_RATE':'OPP_FTA_RATE',
#             'HOME_OREB_PCT':'TEAM_OREB_PCT',
#             'AWAY_OREB_PCT':'OPP_OREB_PCT',
#             'HOME_NET_COMPOSITE_EFFORT':'TEAM_NET_COMPOSITE_EFFORT',
#             'AWAY_NET_COMPOSITE_EFFORT':'OPP_NET_COMPOSITE_EFFORT'
#         })
#         .assign(HOME_IND=1)
#     )

#     df_away = (
#         df[cols]
#         .rename(columns={
#             'AWAY_TEAM':'TEAM',
#             'HOME_TEAM':'OPP_TEAM',
#             'HOME_EFG_PCT':'OPP_EFG_PCT',
#             'AWAY_EFG_PCT':'TEAM_EFG_PCT',
#             'HOME_TM_TOV_PCT': 'OPP_TOV_PCT',
#             'AWAY_TM_TOV_PCT':'TEAM_TM_TOV_PCT',
#             'HOME_FTA_RATE':'OPP_FTA_RATE',
#             'AWAY_FTA_RATE':'TEAM_FTA_RATE',
#             'HOME_OREB_PCT':'OPP_OREB_PCT',
#             'AWAY_OREB_PCT':'TEAM_OREB_PCT',
#             'HOME_NET_COMPOSITE_EFFORT':'OPP_NET_COMPOSITE_EFFORT',
#             'AWAY_NET_COMPOSITE_EFFORT':'TEAM_NET_COMPOSITE_EFFORT'
#         })
#         .assign(HOME_IND=0)
#     )
#     team_schedule = (
#         pd.concat([df_home, df_away], ignore_index=True)
#         .sort_values(['TEAM','GAME_DATE'])
#         .reset_index(drop=True)
#     )

#     # Create season for grouping
#     team_schedule['GAME_ID_str'] = team_schedule['GAME_ID'].astype(str)
#     team_schedule['SEASON_SUFFIX'] = team_schedule['GAME_ID_str'].str[1:3]
#     team_schedule['SEASON'] = '20' + team_schedule['SEASON_SUFFIX']
#     team_schedule['SEASON'] = team_schedule['SEASON'].astype(int)
#     team_schedule.drop(['GAME_ID_str','SEASON_SUFFIX'], axis=1, inplace=True)
#     team_schedule.drop(['OPP_EFG_PCT','OPP_TOV_PCT','OPP_FTA_RATE','OPP_OREB_PCT','OPP_NET_COMPOSITE_EFFORT'], axis=1, inplace=True)

#     return team_schedule.sort_values(['TEAM','GAME_DATE'])


def run_stage2(split_date: str, window: int):
    DATA_DIR = 'data/'
    transformed_path = os.path.join(DATA_DIR, 'transformed_data', 'df_transformed.csv')
    df_trans = pd.read_csv(transformed_path)
    stage1_path = os.path.join(DATA_DIR, 'stage1_effort', 'df_net_stage1_effort.csv')
    df_stage1 = pd.read_csv(stage1_path)

    # Transform Data for Stage 2 Model
    trans_data = TransformStage2(
        df_transformed=df_trans,
        df_stage1_results=df_stage1,
        split_date=split_date,
        hist_avg_window=window
    )
    trans_data.run_transform()

    # Run model
    target = 'EST_HOME_NRtg'
    X_cols = [
        f'AVG{window}_EFG_PCT_DIFF',
        f'AVG{window}_FTA_RATE_DIFF',
        f'AVG{window}_TM_TOV_PCT_DIFF',
        f'AVG{window}_OREB_PCT_DIFF',
        f'AVG{window}_NET_COMPOSITE_EFFORT_DIFF'
    ]
    y = trans_data.df_train[target]
    X = trans_data.df_train[X_cols]
    X1 = sm.add_constant(X)
    reg_stage2 = sm.OLS(endog=y, exog=X1).fit()
    print(reg_stage2.summary())

    # Plots/Diagnostics
    df_c = pd.concat([y,X],axis=1,ignore_index=True)
    df_c.columns=['EST_HOME_NRtg']+[f'AVG{window}_EFG_PCT_DIFF',f'AVG{window}_FTA_RATE_DIFF',f'AVG{window}_TM_TOV_PCT_DIFF',f'AVG{window}_OREB_PCT_DIFF',f'AVG{window}_NET_COMPOSITE_EFFORT_DIFF']
    print(df_c.describe(percentiles=[.001,.01,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99,.999]))
    sns.pairplot(df_c)
    plt.show()
    yhat = reg_stage2.fittedvalues
    res = reg_stage2.resid
    sns.scatterplot(x=yhat, y=res)
    plt.axhline(y=0, linestyle='dashed',color='red')
    plt.show()
    sm.qqplot(res, line='q')
    plt.show()
    plot_data = pd.DataFrame({'y_true': y, 'y_pred': yhat})
    sns.regplot(
            x='y_true', y='y_pred', data=plot_data, 
            line_kws={'color': 'red', 'label':'Linear Model'}, 
            scatter_kws={'color': 'tab:blue','alpha':.25}
        )
    sns.regplot(
            x='y_true', y='y_pred', data=plot_data, 
            lowess=True,
            line_kws={'color': 'green', 'label':'Lowess'}, 
            scatter_kws={'color': 'tab:blue','alpha':.25}
        )
    plt.legend(loc='best')
    plt.show()

if __name__=='__main__':
    run_stage2(split_date='2024-10-22',window=10)