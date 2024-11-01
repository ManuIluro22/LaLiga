import pandas as pd
import numpy as np


class QuinielaDataframe:

    def __init__(self,dataframe):
            
            self.df = dataframe
            self._clean_data()
            self._generate_team_info()

    def _clean_data(self):

        self.df[['home_score', 'away_score']] = self.df['score'].str.split(':', expand=True)
        self.df['home_score'] = pd.to_numeric(self.df['home_score'])
        self.df['away_score'] = pd.to_numeric(self.df['away_score'])

        self.df['result'] = self.df.apply(lambda row: '1' if row['home_score'] > row['away_score']
                                        else '2' if row['home_score'] < row['away_score']
                                        else 'X', axis=1)
    
        self.df['date'] = pd.to_datetime(self.df['date'], format='%m/%d/%y')
        self.df['weekday'] = self.df['date'].dt.day_name()
        self.df['weekday_num'] = self.df['date'].dt.dayofweek
    

    def generate_features(self,list_features=None):

        self._last_n_matchdays()
        
    
    def _last_n_matchdays(self,n=5):

        self.team_results['result'] = self.team_results.apply(lambda row: 'W' if row['W'] == 1 else 'L' if row['L'] == 1 else 'T', axis=1)
        self.team_results.sort_values(['season', 'division', 'team', 'matchday'],inplace = True)

        last_results = self.team_results.groupby(['season', 'division', 'team'])['result'].apply(lambda x: np.array([np.array(rolling_list) for rolling_list in x.rolling(n)],dtype=object))
        last_results = np.concatenate(last_results.values).reshape(-1)
        
        self.match_standings.sort_values(['season', 'division', 'team', 'matchday'],inplace = True)

        self.match_standings.reset_index(inplace=True)

        self.match_standings["avg_last_points"] = pd.Series(last_results).apply(self._calculate_avg_points)

    def _calculate_avg_points(self,last_n):
        
        points = {'W': 3, 'T': 1, 'L': 0}
        
        total_points = np.mean([points[res] for res in last_n if res in points])
        
        return total_points if len(last_n) > 0 else 0


    def _generate_team_info(self):
        
        self._encoding_win()
        self._generate_matchday_standings()

    def _encoding_win(self):

        self.df['home_W'] = (self.df['result'] == '1').astype(int)
        self.df['home_L'] = (self.df['result'] == '2').astype(int)
        self.df['home_T'] = (self.df['result'] == 'X').astype(int)
        self.df['away_W'] = (self.df['result'] == '2').astype(int)
        self.df['away_L'] = (self.df['result'] == '1').astype(int)
        self.df['away_T'] = (self.df['result'] == 'X').astype(int)

    def _generate_matchday_standings(self):
        
        home_df = self.df[['season', 'division', 'matchday', 'home_team', 'home_score', 'away_score', 'home_score', 'away_score', 'home_W', 'home_L', 'home_T']]
        home_df.columns = ['season', 'division', 'matchday', 'team', 'GF', 'GA', 'home_GF', 'home_GA', 'W', 'L', 'T']

        away_df = self.df[['season', 'division', 'matchday', 'away_team', 'away_score', 'home_score', 'away_score', 'home_score', 'away_W', 'away_L', 'away_T']]
        away_df.columns = ['season', 'division', 'matchday', 'team', 'GF', 'GA', 'away_GF', 'away_GA' ,'W', 'L', 'T']

        self.team_results = pd.concat([home_df, away_df])
        self.team_results.fillna(0,inplace=True)

        self.match_standings = self.team_results.sort_values(['season', 'division', 'team', 'matchday'])
        self.match_standings[['GF', 'GA','home_GF','home_GA','away_GF','away_GA', 'W', 'L', 'T']] = self.match_standings.groupby(['season', 'division', 'team'])[['GF', 'GA','home_GF','home_GA','away_GF','away_GA', 'W', 'L', 'T']].cumsum()

        self.match_standings['GD'] = self.match_standings['GF'] - self.match_standings['GA']
        self.match_standings['M'] = self.match_standings['W'] + self.match_standings['L'] + self.match_standings['T']
        self.match_standings['Pts'] = self.match_standings['W'] * 3 + self.match_standings['T']

        self.match_standings = self.match_standings.sort_values(by=['season','division', 'matchday',
                                                  'Pts', 'GD', 'GF'],
                                              ascending=[True, True,True,
                                                         False, False, False])

        self.match_standings.insert(3,'rank',self.match_standings.groupby(['season','division',
                                                   'matchday'])['Pts'].rank(ascending=False,
                                                                            method='first'))





