import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import four_factors_stats
from src.steps.transform.transform_stage2 import TransformStage2

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

    # Prep columns and target
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
    #X1 = sm.add_constant(X)

    # Fit Model
    #reg_stage2 = sm.OLS(endog=y, exog=X1).fit()
    #print(reg_stage2.summary())

    # Plots/Diagnostics
    # df_c = pd.concat([y,X],axis=1,ignore_index=True)
    # df_c.columns=['EST_HOME_NRtg']+[f'AVG{window}_EFG_PCT_DIFF',f'AVG{window}_FTA_RATE_DIFF',f'AVG{window}_TM_TOV_PCT_DIFF',f'AVG{window}_OREB_PCT_DIFF',f'AVG{window}_NET_COMPOSITE_EFFORT_DIFF']
    # print(df_c.describe(percentiles=[.001,.01,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99,.999]))
    # sns.pairplot(df_c)
    # plt.show()
    # yhat = reg_stage2.fittedvalues
    # res = reg_stage2.resid
    # sns.scatterplot(x=yhat, y=res)
    # plt.axhline(y=0, linestyle='dashed',color='red')
    # plt.show()
    # sm.qqplot(res, line='q')
    # plt.show()
    # plot_data = pd.DataFrame({'y_true': y, 'y_pred': yhat})
    # sns.regplot(
    #         x='y_true', y='y_pred', data=plot_data, 
    #         line_kws={'color': 'red', 'label':'Linear Model'}, 
    #         scatter_kws={'color': 'tab:blue','alpha':.25}
    #     )
    # sns.regplot(
    #         x='y_true', y='y_pred', data=plot_data, 
    #         lowess=True,
    #         line_kws={'color': 'green', 'label':'Lowess'}, 
    #         scatter_kws={'color': 'tab:blue','alpha':.25}
    #     )
    # plt.legend(loc='best')
    # plt.show()

if __name__=='__main__':
    run_stage2(split_date='2024-10-22',window=10)