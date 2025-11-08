import pandas as pd

class TransformStage3:
    def __init__(self, df_transformed: pd.DataFrame, df_stage1_output: pd.DataFrame, df_train_stage2_output: pd.DataFrame, current_date: str) -> None:
        self.df_transformed = df_transformed
        self.df_stage1_output = df_stage1_output
        self.df_train_stage2_output = df_train_stage2_output
        self.current_date = pd.to_datetime(current_date, errors='coerce')

        self.stage1_cols = ['GAME_ID','TEAM_ID','NET_COMPOSITE_EFFORT']
        

    def merge_to_stage2_output(self) -> None:
        # Merge home net effort
        self.df_train_stage2_output = self.df_train_stage2_output.merge(
            self.df_stage1_output[self.stage1_cols],
            how='left',
            left_on=['GAME_ID','HOME_TEAM_ID'],
            right_on=['GAME_ID','TEAM_ID']
        )
        self.df_train_stage2_output.drop('TEAM_ID', axis=1, inplace=True)
        self.df_train_stage2_output.rename(columns={'NET_COMPOSITE_EFFORT': 'HOME_NET_COMPOSITE_EFFORT'}, inplace=True)

        # Merge away net effort
        self.df_train_stage2_output = self.df_train_stage2_output.merge(
            self.df_stage1_output[self.stage1_cols],
            how='left', 
            left_on=['GAME_ID','AWAY_TEAM_ID'],
            right_on=['GAME_ID','TEAM_ID']
        )
        self.df_train_stage2_output.drop('TEAM_ID', axis=1, inplace=True)
        self.df_train_stage2_output.rename(columns={'NET_COMPOSITE_EFFORT': 'AWAY_NET_COMPOSITE_EFFORT'}, inplace=True)

    def clean_transformed_data(self):
        """Clean Columns and Split Data"""
        end_home_cols = self.df_transformed.columns[self.df_transformed.columns.str.endswith('_HOME')]
        end_away_cols = self.df_transformed.columns[self.df_transformed.columns.str.endswith('_AWAY')]
        
        if len(end_home_cols) > 0 or len(end_away_cols) > 0:
            begin_home_cols = [f"HOME_{col.replace('_HOME','')}" for col in end_home_cols]
            begin_away_cols = [f"AWAY_{col.replace('_AWAY','')}" for col in end_away_cols]
            home_cols = dict(zip(end_home_cols, begin_home_cols))
            away_cols = dict(zip(end_away_cols, begin_away_cols))
            new_cols = {**home_cols, **away_cols}
            self.df_transformed.rename(columns=new_cols, inplace=True)

        self.df_transformed.drop(['HOME_TEAM_NAME','AWAY_TEAM_NAME'], axis=1, inplace=True)
        self.df_transformed.rename(
            columns={
                'HOME_TEAM_ABBREVIATION':'HOME_TEAM',
                'AWAY_TEAM_ABBREVIATION':'AWAY_TEAM'
            }, 
            inplace=True
        )

        self.df_transformed['GAME_DATE'] = pd.to_datetime(self.df_transformed['GAME_DATE'])
        
        self.df_transformed = self.df_transformed[self.df_transformed['GAME_DATE'] <= self.current_date]

    # def merge_target_to_stage2_output(self):
    #     self.df_train_stage2_output = self.df_train_stage2_output.merge(
    #         self.df_transformed[['GAME_ID','EST_HOME_NRtg']],
    #         how='left',
    #         on='GAME_ID'
    #     )

    def run_transform(self):
        # Step 1 - Join Stage 1 Effort to Stage 2 Output
        self.merge_to_stage2_output()

        # Step 2 - Clean Transformed Data (is this even necessary?)
        self.clean_transformed_data()

        # Step 3 - Join Net Rating to Stage 2 Output
        #self.merge_target_to_stage2_output()

        #print(self.df_train_stage2_output.head())
        #print(self.df_train_stage2_output.tail())
        #print(self.df_train_stage2_output.columns)
