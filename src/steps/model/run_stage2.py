import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import four_factors_stats

def team_schedule(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'GAME_ID','HOME_TEAM','AWAY_TEAM','GAME_DATE',
        'HOME_EFG_PCT','AWAY_EFG_PCT', 
        'HOME_TM_TOV_PCT','AWAY_TM_TOV_PCT',
        'HOME_FTA_RATE','AWAY_FTA_RATE',
        'HOME_OREB_PCT','AWAY_OREB_PCT',
        'HOME_NET_COMPOSITE_EFFORT','AWAY_NET_COMPOSITE_EFFORT'
    ]
    df_home = (
        df[cols]
        .rename(columns={
            'HOME_TEAM':'TEAM',
            'AWAY_TEAM':'OPP_TEAM',
            'HOME_EFG_PCT':'TEAM_EFG_PCT',
            'AWAY_EFG_PCT':'OPP_EFG_PCT',
            'HOME_TM_TOV_PCT': 'TEAM_TM_TOV_PCT',
            'AWAY_TM_TOV_PCT':'OPP_TOV_PCT',
            'HOME_FTA_RATE':'TEAM_FTA_RATE',
            'AWAY_FTA_RATE':'OPP_FTA_RATE',
            'HOME_OREB_PCT':'TEAM_OREB_PCT',
            'AWAY_OREB_PCT':'OPP_OREB_PCT',
            'HOME_NET_COMPOSITE_EFFORT':'TEAM_NET_COMPOSITE_EFFORT',
            'AWAY_NET_COMPOSITE_EFFORT':'OPP_NET_COMPOSITE_EFFORT'
        })
        .assign(HOME_IND=1)
    )

    df_away = (
        df[cols]
        .rename(columns={
            'AWAY_TEAM':'TEAM',
            'HOME_TEAM':'OPP_TEAM',
            'HOME_EFG_PCT':'OPP_EFG_PCT',
            'AWAY_EFG_PCT':'TEAM_EFG_PCT',
            'HOME_TM_TOV_PCT': 'OPP_TOV_PCT',
            'AWAY_TM_TOV_PCT':'TEAM_TM_TOV_PCT',
            'HOME_FTA_RATE':'OPP_FTA_RATE',
            'AWAY_FTA_RATE':'TEAM_FTA_RATE',
            'HOME_OREB_PCT':'OPP_OREB_PCT',
            'AWAY_OREB_PCT':'TEAM_OREB_PCT',
            'HOME_NET_COMPOSITE_EFFORT':'OPP_NET_COMPOSITE_EFFORT',
            'AWAY_NET_COMPOSITE_EFFORT':'TEAM_NET_COMPOSITE_EFFORT'
        })
        .assign(HOME_IND=0)
    )
    team_schedule = (
        pd.concat([df_home, df_away], ignore_index=True)
        .sort_values(['TEAM','GAME_DATE'])
        .reset_index(drop=True)
    )

    # Create season for grouping
    team_schedule['GAME_ID_str'] = team_schedule['GAME_ID'].astype(str)
    team_schedule['SEASON_SUFFIX'] = team_schedule['GAME_ID_str'].str[1:3]
    team_schedule['SEASON'] = '20' + team_schedule['SEASON_SUFFIX']
    team_schedule['SEASON'] = team_schedule['SEASON'].astype(int)
    team_schedule.drop(['GAME_ID_str','SEASON_SUFFIX'], axis=1, inplace=True)
    team_schedule.drop(['OPP_EFG_PCT','OPP_TOV_PCT','OPP_FTA_RATE','OPP_OREB_PCT','OPP_NET_COMPOSITE_EFFORT'], axis=1, inplace=True)

    return team_schedule.sort_values(['TEAM','GAME_DATE'])


def run_stage2(split_date: str, window: int):
    DATA_DIR = 'data/'
    transformed_path = os.path.join(DATA_DIR, 'transformed_data', 'df_transformed.csv')
    df_trans = pd.read_csv(transformed_path)
    stage1_path = os.path.join(DATA_DIR, 'stage1_effort', 'df_net_stage1_effort.csv')
    df_stage1 = pd.read_csv(stage1_path)

    # Merge stage 1 results onto transformed data
    # In two stages: 1. Merge home results, 2. Merge away results
    df_trans = df_trans.merge(
        df_stage1.drop(['SEASON_ID','GAME_DATE','TEAM_ABBREVIATION'],axis=1),
        how='left',
        left_on=['GAME_ID','HOME_TEAM_ID'],
        right_on=['GAME_ID','TEAM_ID']
    )
    df_trans = df_trans.merge(
        df_stage1.drop(['SEASON_ID','GAME_DATE','TEAM_ABBREVIATION'],axis=1),
        how='left',
        left_on=['GAME_ID','AWAY_TEAM_ID'],
        right_on=['GAME_ID','TEAM_ID'],
        suffixes=('_HOME','_AWAY')
    )

    # Clean up columns
    end_home_cols = df_trans.columns[df_trans.columns.str.endswith('_HOME')]
    end_away_cols = df_trans.columns[df_trans.columns.str.endswith('_AWAY')]
    begin_home_cols = [f"HOME_{col.replace('_HOME','')}" for col in end_home_cols]
    begin_away_cols = [f"AWAY_{col.replace('_AWAY','')}" for col in end_away_cols]
    home_cols = dict(zip(end_home_cols, begin_home_cols))
    away_cols = dict(zip(end_away_cols, begin_away_cols))
    new_cols = {**home_cols, **away_cols}
    df_trans.rename(columns=new_cols, inplace=True)

    df_trans.drop(['HOME_TEAM_NAME','AWAY_TEAM_NAME'], axis=1, inplace=True)
    df_trans.rename(columns={'HOME_TEAM_ABBREVIATION':'HOME_TEAM','AWAY_TEAM_ABBREVIATION':'AWAY_TEAM'}, inplace=True)

    df_trans['GAME_DATE'] = pd.to_datetime(df_trans['GAME_DATE'])
    df_train = df_trans[df_trans['GAME_DATE'] < split_date].copy()

    # Create features
    # Need historical four factors average and historical effort average
    df_team_sched = team_schedule(df=df_train)
    df_team_sched[f'TEAM_AVG{window}_NET_COMPOSITE_EFFORT'] = (
        df_team_sched
        .groupby('TEAM')['TEAM_NET_COMPOSITE_EFFORT']
        .rolling(window=window, min_periods=1, closed='left')
        .mean()
        .reset_index(drop=True)
    )
    for ff in four_factors_stats:
        df_team_sched[f"TEAM_AVG{window}_{ff}"] = (
            df_team_sched
            .groupby('TEAM')[f'TEAM_{ff}']
            .rolling(window=window, min_periods=1, closed='left')
            .mean()
            .reset_index(drop=True)
        )

    # Join back to transformed data: 1. join by home, 2. join by away 
    # Only retain necessary columns
    df_team_sched = df_team_sched[['GAME_ID','TEAM','OPP_TEAM',
                                   f'TEAM_AVG{window}_NET_COMPOSITE_EFFORT',
                                   f'TEAM_AVG{window}_EFG_PCT',f'TEAM_AVG{window}_FTA_RATE',
                                   f'TEAM_AVG{window}_TM_TOV_PCT',f'TEAM_AVG{window}_OREB_PCT']]
    df_train = (
        df_train
        .merge(
        df_team_sched, 
        how='left', 
        left_on=['GAME_ID','HOME_TEAM'],
        right_on=['GAME_ID','TEAM'])
        .rename(columns={f'TEAM_AVG{window}_NET_COMPOSITE_EFFORT':f'HOME_AVG{window}_NET_COMPOSITE_EFFORT',
                        f'TEAM_AVG{window}_EFG_PCT':f'HOME_AVG{window}_EFG_PCT',
                        f'TEAM_AVG{window}_FTA_RATE':f'HOME_AVG{window}_FTA_RATE',
                        f'TEAM_AVG{window}_TM_TOV_PCT':f'HOME_AVG{window}_TM_TOV_PCT',
                        f'TEAM_AVG{window}_OREB_PCT':f'HOME_AVG{window}_OREB_PCT'})
        .drop(['TEAM','OPP_TEAM'],axis=1)
    )
    df_train = (
        df_train
        .merge(
            df_team_sched,
            how='left',
            left_on=['GAME_ID','AWAY_TEAM'],
            right_on=['GAME_ID','TEAM']
        )
        .rename(columns={f'TEAM_AVG{window}_NET_COMPOSITE_EFFORT':f'AWAY_AVG{window}_NET_COMPOSITE_EFFORT',
                        f'TEAM_AVG{window}_EFG_PCT':f'AWAY_AVG{window}_EFG_PCT',
                        f'TEAM_AVG{window}_FTA_RATE':f'AWAY_AVG{window}_FTA_RATE',
                        f'TEAM_AVG{window}_TM_TOV_PCT':f'AWAY_AVG{window}_TM_TOV_PCT',
                        f'TEAM_AVG{window}_OREB_PCT':f'AWAY_AVG{window}_OREB_PCT'})
        .drop(['TEAM','OPP_TEAM'],axis=1)
        
    )
    df_train[f'AVG{window}_NET_COMPOSITE_EFFORT_DIFF'] = df_train[f'HOME_AVG{window}_NET_COMPOSITE_EFFORT']-df_train[f'AWAY_AVG{window}_NET_COMPOSITE_EFFORT']
    df_train[f'AVG{window}_EFG_PCT_DIFF'] = df_train[f'HOME_AVG{window}_EFG_PCT']-df_train[f'AWAY_AVG{window}_EFG_PCT']
    df_train[f'AVG{window}_FTA_RATE_DIFF'] = df_train[f'HOME_AVG{window}_FTA_RATE']-df_train[f'AWAY_AVG{window}_FTA_RATE']
    df_train[f'AVG{window}_TM_TOV_PCT_DIFF'] = df_train[f'HOME_AVG{window}_TM_TOV_PCT']-df_train[f'AWAY_AVG{window}_TM_TOV_PCT']
    df_train[f'AVG{window}_OREB_PCT_DIFF'] = df_train[f'HOME_AVG{window}_OREB_PCT']-df_train[f'AWAY_AVG{window}_OREB_PCT']

    df_train.fillna(0, inplace=True)

    # Run model
    y = df_train['EST_HOME_NRtg']
    X = df_train[[f'AVG{window}_EFG_PCT_DIFF',f'AVG{window}_FTA_RATE_DIFF',f'AVG{window}_TM_TOV_PCT_DIFF',f'AVG{window}_OREB_PCT_DIFF',f'AVG{window}_NET_COMPOSITE_EFFORT_DIFF']]
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