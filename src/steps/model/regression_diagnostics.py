import os
from typing import Dict
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

class RegressionDiagnostics:
    def plot_fit_vs_actual(self, y_true, y_pred, ax, target):
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
        ax.set_ylabel(f"Est. {target}")
        ax.set_xlabel(f"Actual {target}")
        ax.legend(loc='best')

    def plot_residuals_vs_fit(self, y_true, y_pred, target, ax):
        """Plot residuals vs predicted values"""
        residuals = y_true.flatten()-y_pred.flatten()
        sns.scatterplot(
            x=y_pred,
            y=residuals,
            alpha=.5, 
            ax=ax
        )
        ax.axhline(y=0, color='red', linestyle='dashed')
        ax.set_ylabel('Residual')
        ax.axhline(y=0, color='red',linestyle='dashed')
        ax.set_ylabel('Residual')
        ax.set_xlabel(f'Est. {target}')

    def plot_qq(self,y_true, y_pred, ax):
        residuals = y_true.flatten()-y_pred.flatten()
        sm.qqplot(residuals, line='q', ax=ax)

    def plot_error_analysis(self, y_pred, y_true, axes):
        """Create error analysis plots"""
        df_res = self._create_results_dataframe(y_pred, y_true)

        stats_dict = self._calculate_binned_statistics(df_res)

        self._plot_binned_statistics(axes, stats_dict)

    def _create_results_dataframe(self, y_pred, y_true):
        residuals = y_true.flatten()-y_pred.flatten()
        df_res = pd.DataFrame(
            {
                'yhat': y_pred,
                'y': y_true,
                'resid': residuals,
                'abs_resid': abs(residuals)
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

    def create_diagnostic_plots(self,model_name: str, target: str, y_true, y_pred, save_figs=False):
        """Produce Diagnostic Plots"""
        fig, ax = plt.subplots(2,3, figsize=(20,10))
        plt.suptitle(f'{model_name} Diagnositic Plots')

        self.plot_fit_vs_actual(y_true=y_true, y_pred=y_pred, ax=ax[0,0], target=target)
        self.plot_residuals_vs_fit(y_true=y_true, y_pred=y_pred, target=target, ax=ax[0,1])
        self.plot_qq(y_true=y_true, y_pred=y_pred,ax=ax[0,2])

        self.plot_error_analysis(y_true=y_true, y_pred=y_pred, axes=ax[1,:])

        if save_figs:
            os.makedirs('figures',exist_ok=True)
            plt.savefig(f"figures/{model_name}_model_diagnositic_plots.png")

        plt.show()