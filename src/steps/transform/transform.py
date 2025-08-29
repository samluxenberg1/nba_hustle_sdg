import pandas as pd
import numpy as np

class TransformData:
    def __init__(self, df_proc: pd.DataFrame): 
        self.df_proc = df_proc

    def engineered_features(self) -> pd.DataFrame:
        
        # Estimate number of possessions = FGA + (0.44 x FTA) + TOV - OREB
        self.df_proc['HOME_POSS'] = self.df_proc['HOME_FGA'] + .44*self.df_proc['HOME_FTA'] + self.df_proc['HOME_TOV'] - self.df_proc['HOME_OREB']
        self.df_proc['AWAY_POSS'] = self.df_proc['AWAY_FGA'] + .44*self.df_proc['AWAY_FTA'] + self.df_proc['AWAY_TOV'] - self.df_proc['AWAY_OREB']

        # Compute Offensive Ratings = PTS / POSS x 100
        self.df_proc['HOME_ORtg'] = self.df_proc['HOME_PTS']/self.df_proc['HOME_POSS']*100
        self.df_proc['AWAY_ORtg'] = self.df_proc['AWAY_PTS']/self.df_proc['AWAY_POSS']*100

        # Compute Net Ratings: 2 equivalent calculations
        # 1. HOME ORtg - HOME DRtg 
        # 2. HOME ORtg - AWAY ORtg
        self.df_proc['HOME_NRtg'] = self.df_proc['HOME_ORtg'] - self.df_proc['AWAY_ORtg']
        self.df_proc['AWAY_NRtg'] = -self.df_proc['HOME_NRtg']

        return self.df_proc