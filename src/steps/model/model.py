import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, Dict, Union, Literal
from statsmodels.regression.linear_model import RegressionResultsWrapper

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.steps.evaluation.utils_evaluation import rmse

class CompositeEffort:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.reg_stage1: Optional[RegressionResultsWrapper] = None

    def fit_stage1_model(self):
        """Fit Stage 1 OLS Model"""
        X1 = sm.add_constant(self.X)
        self.reg_stage1 = sm.OLS(endog=self.y, exog=X1).fit()
        return self.reg_stage1
    
    def _print_model_summary(self):
        """Print Model Summary and RMSE"""
        if self.reg_stage1 is None:
            raise ValueError("Stage 1 model must be fitted first")
        
        print(self.reg_stage1.summary())
        
        # Estimate RMSE
        yhat = self.reg_stage1.fittedvalues
        print(f"\n\nIn-Sample RMSE: {rmse(self.y, yhat): .3f}\n\n")

    def _create_diagnostic_plots(self):
        """Create diagnostic plots for stage 1 model"""
        if self.reg_stage1 is None:
            raise ValueError("Stage 1 model must be fitted first")
        
        fig, ax = plt.subplots(2,3, figsize=(20,10))

        # Fit vs Actual
        self._plot_fit_vs_actual(ax=ax[0,0], y_true=self.y, y_pred=self.reg_stage1.fittedvalues)

        # Residual Plot
        self._plot_residuals_vs_fit(ax=ax[0,1])

        # QQ Plot
        sm.qqplot(self.reg_stage1.resid, line='q', ax=ax[0,2])

        # Error analysis plots
        self._plot_error_analysis(axes=ax[1,:])
    
    def _plot_fit_vs_actual(self, ax, y_true, y_pred):
        """Plot fitted vs actual values"""
        plot_data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

        sns.regplot(
            x='y_true', y='y_pred', data=plot_data, 
            line_kws={'color': 'red', 'label':'Linear Model'}, 
            scatter_kws={'color': 'tab:blue','alpha':.25},
            ax=ax
        )
        sns.regplot(
            x='y_true', y='y_pred', data=plot_data, 
            lowess=True, 
            line_kws={'color': 'green', 'label':'Lowess Model'}, 
            scatter_kws={'color':'blue', 'alpha':0}, 
            ax=ax
        )
        ax.set_ylabel('Estimated NRtg')
        ax.legend(loc='best')

    def _plot_residuals_vs_fit(self,ax):
        """Plot residuals vs predicted values"""
        if self.reg_stage1 is None:
            raise ValueError("Stage 1 model must be fitted first")
        
        sns.scatterplot(
            x=self.reg_stage1.fittedvalues, 
            y=self.reg_stage1.resid, 
            alpha=.5, 
            ax=ax
        )
        ax.axhline(y=0, color='red',linestyle='dashed')
        ax.set_ylabel('Residual')
        ax.set_xlabel('Predicted NRtg')

    def _plot_error_analysis(self, axes):
        """Create error analysis plots"""
        df_res = self._create_results_dataframe()

        stats_dict = self._calculate_binned_statistics(df_res)

        self._plot_binned_statistics(axes, stats_dict)

    def _create_results_dataframe(self):
        if self.reg_stage1 is None:
            raise ValueError("Stage 1 model must be fitted first")
        
        df_res = pd.DataFrame(
            {
                'yhat': self.reg_stage1.fittedvalues,
                'y': self.y,
                'resid': self.reg_stage1.resid,
                'abs_resid': abs(self.reg_stage1.resid)
            }
        )

        # Create bins
        df_res['yhat_bin'] = pd.qcut(df_res['yhat'], q=10)
        df_res['y_bin'] = pd.qcut(df_res['y'], q=10)

        return df_res
    
    def _calculate_binned_statistics(self, df_res: pd.DataFrame): 
        """Calculate statistics by bin"""
        return {
            'y_mean_per_yhat_bin': df_res.groupby('yhat_bin', observed=True)['y'].mean(),
            'yhat_mean_per_y_bin': df_res.groupby('y_bin', observed=True)['yhat'].mean(),
            'resid_mean_per_yhat_bin': df_res.groupby('yhat_bin', observed=True)['resid'].mean(),
            'resid_mean_per_y_bin': df_res.groupby('y_bin', observed=True)['resid'].mean(),
            'abs_resid_mean_per_yhat_bin': df_res.groupby('yhat_bin', observed=True)['abs_resid'].mean(),
            'abs_resid_mean_per_y_bin': df_res.groupby('y_bin', observed=True)['abs_resid'].mean()
        }
    
    def _plot_binned_statistics(self, axes: np.ndarray, stats_dict: Dict): 
        """Plot binned statistics"""
        x_range = range(1,11)

        # Plot 1: Y Means
        sns.lineplot(
            x=x_range, y=stats_dict['y_mean_per_yhat_bin'], 
            label='Avg Y Per Yhat Bin', ax=axes[0]
        )
        sns.lineplot(
            x=x_range, y=stats_dict['yhat_mean_per_y_bin'], 
            label='Avg Yhat Per Y Bin', ax=axes[0]
        )
        axes[0].legend(loc='best')
        
        # Plot 2: Residual Means
        sns.lineplot(
            x=x_range, y=stats_dict['resid_mean_per_yhat_bin'], 
            label='Avg Residual Per Yhat Bin', ax=axes[1]
        )
        sns.lineplot(
            x=x_range, y=stats_dict['resid_mean_per_y_bin'], 
            label='Avg Residual Per Y Bin', ax=axes[1]
        )
        axes[1].legend(loc='best')
        
        # Plot 3: Absolute Residual Means
        sns.lineplot(
            x=x_range, y=stats_dict['abs_resid_mean_per_yhat_bin'], 
            label='Avg Abs. Residual Per Yhat Bin', ax=axes[2]
        )
        sns.lineplot(
            x=x_range, y=stats_dict['abs_resid_mean_per_y_bin'], 
            label='Avg Abs. Residual Per Y Bin', ax=axes[2]
        )
        axes[2].legend(loc='best')

    
    def create_composite_effort(self, off_def: Literal['off','def']):
        """Create composite effort"""
        if self.reg_stage1 is None:
            raise ValueError("Stage 1 model must be fitted first")
        
        effort_vec = np.array(self.reg_stage1.params[1:]).reshape((-1,1))
        effort_mat = np.array(self.X)

        self.X[f'{off_def.upper()}_COMPOSITE_EFFORT'] = effort_mat @ effort_vec

    
    def analyze_composite_effort(self, off_def: Literal['off','def']):
        """Analyze composite effor distribution and correlation"""
        self._plot_composite_effort_analysis(off_def)
        self._print_composite_effort_stats(off_def)

    def _plot_composite_effort_analysis(self, off_def: Literal['off','def']):
        
        plot_data = pd.DataFrame(
            {
                f'{off_def.upper()}_COMPOSITE_EFFORT': self.X[f'{off_def.upper()}_COMPOSITE_EFFORT'],
                'y': self.y
            }
        )
        
        fig, ax = plt.subplots(1,3, figsize=(20,5))
        sns.histplot(x=f'{off_def.upper()}_COMPOSITE_EFFORT', data=plot_data, ax=ax[0])
        sns.ecdfplot(x=f'{off_def.upper()}_COMPOSITE_EFFORT', data=plot_data, ax=ax[1])
        sns.scatterplot(x=f'{off_def.upper()}_COMPOSITE_EFFORT', y='y', data=plot_data, ax=ax[2])

        plt.tight_layout()
        plt.show()
    
    def _print_composite_effort_stats(self, off_def: Literal['off','def']):
        """Print composite effort statistics and correlation"""
        
        percentiles = [.001,.01,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99,.999]
        print(self.X[f'{off_def.upper()}_COMPOSITE_EFFORT'].describe(percentiles))

        # Correlations
        pearson_stat, pearson_pval = stats.pearsonr(self.X[f'{off_def.upper()}_COMPOSITE_EFFORT'],self.y)
        spearman_stat, spearman_pval = stats.spearmanr(self.X[f'{off_def.upper()}_COMPOSITE_EFFORT'],self.y)
        kendall_stat, kendall_pval = stats.kendalltau(self.X[f'{off_def.upper()}_COMPOSITE_EFFORT'],self.y)

        print(f"\n\nPearson Correlation: {pearson_stat: .3f}, P-value: {pearson_pval: .5f}")
        print(f"Spearman Correlation: {spearman_stat: .3f}, P-value: {spearman_pval: .5f}")
        print(f"Kendall's Tau Correlation: {kendall_stat: .3f}, P-value: {kendall_pval: .5f}")

    def estimate_stage1_model(self):
        """Complete stage 1 estiamtation workflow"""
        self.fit_stage1_model()
        self._print_model_summary()
        self._create_diagnostic_plots()

        return self.reg_stage1
    
    def estimate_composite_effort(self, off_def: Literal['off','def'], output_dir: str):
        """Complete composite effort workflow"""
        self.create_composite_effort(off_def)
        self.analyze_composite_effort(off_def)

        # Save data
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'df_{off_def}_stage1_effort.csv')
        self.X.to_csv(output_path, index=False)

        return self.X

if __name__=='__main__':
    from src.constants import hustle_stats
    # Read in data
    DATA_DIR = 'data/'
    input_path = os.path.join(DATA_DIR, 'transformed_data', 'df_transformed.csv')
    df_trans = pd.read_csv(input_path)

    # Features
    features_to_exclude = ['CONTESTED_SHOTS','BOX_OUTS', 'SCREEN_AST_PTS', 'BOX_OUT_PLAYER_TEAM_REBS', 'LOOSE_BALLS_RECOVERED','BOX_OUT_PLAYER_REBS']
    features = list(set(hustle_stats)-set(features_to_exclude))
    home_features = [f"HOME_{feat}" for feat in features]
    away_features = [f"AWAY_{feat}" for feat in features]
    features = home_features + away_features
    X = df_trans[features]
    y = df_trans['HOME_NRtg']

    comp_eff = CompositeEffort(X=X, y=y)
    model = comp_eff.estimate_stage1_model()

    #output_dir = os.path.join(DATA_DIR, 'stage1_effort')
    #df_stage1_effort = comp_eff.estimate_composite_effort(output_dir=output_dir)

    print(f"Model: {model}")
    #print(df_stage1_effort.head())