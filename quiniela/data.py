import pandas as pd
import numpy as np

from feature_generation import FeatureGenerator

class QuinielaDataframe:

    def __init__(self, dataframe):

        self.df = dataframe
        self._clean_data()
        self._generate_team_info()

    def generate_features(self, list_features=[]):

        feature_generator = FeatureGenerator()
        feature_generator.apply_features(self, list_features)

    def generate_matchday_dataframe(self):
        
        return MatchdayDataframe(self)
    
    
    def _clean_data(self):

        self.df[['home_score', 'away_score']] = self.df['score'].str.split(':', expand=True)
        self.df['home_score'] = pd.to_numeric(self.df['home_score'])
        self.df['away_score'] = pd.to_numeric(self.df['away_score'])
        
        self.df['result'] = self.df.apply(lambda row: '1' if row['home_score'] > row['away_score']
                                          else '2' if row['home_score'] < row['away_score']
                                          else 'X', axis=1)
        
        self.df["season"] = pd.to_numeric(self.df["season"].str[-4:])

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

        home_df = self.df[['season', 'division', 'matchday', 'home_team', 'home_score', 'away_score','home_score', 'away_score', 'home_W', 'home_L', 'home_T']]
        home_df.columns = ['season', 'division', 'matchday', 'team', 'GF', 'GA','home_GF', 'home_GA', 'W', 'L', 'T']
        
        away_df = self.df[['season', 'division', 'matchday', 'away_team', 'away_score', 'home_score','away_score', 'home_score', 'away_W', 'away_L', 'away_T']]
        away_df.columns = ['season', 'division', 'matchday', 'team', 'GF', 'GA','away_GF', 'away_GA', 'W', 'L', 'T']
        
        self.team_results = pd.concat([home_df, away_df])
        self.team_results.fillna(0, inplace=True)

        self.match_standings = self.team_results.sort_values(['season', 'division', 'team', 'matchday'])
        self.match_standings[['GF', 'GA','home_GF','home_GA','away_GF','away_GA', 'W', 'L', 'T']] = self.match_standings.groupby(['season', 'division', 'team'])[['GF', 'GA','home_GF','home_GA','away_GF','away_GA', 'W', 'L', 'T']].cumsum()

        self.match_standings['GD'] = self.match_standings['GF'] - self.match_standings['GA']
        self.match_standings['M'] = self.match_standings['W'] + self.match_standings['L'] + self.match_standings['T']
        self.match_standings['Pts'] = self.match_standings['W'] * 3 + self.match_standings['T']

        self.match_standings = self.match_standings.sort_values(by=['season', 'division', 'matchday', 'Pts', 'GD', 'GF'],
                                                                ascending=[True, True, True, False, False, False])

        self.match_standings.insert(3, 'rank', self.match_standings.groupby(['season', 'division', 'matchday'])['Pts']
                                    .rank(ascending=False, method='first'))


class MatchdayDataframe:
    
    def __init__(self, quiniela_df,versus_features = True):
        
        self.df = quiniela_df.df.copy()
        self.matchday_df = quiniela_df.match_standings.copy()
        self._prepare_matchday_stats()

        if(versus_features):
            feature_generator = FeatureGenerator()
            feature_generator.apply_features(self, ['versus_features'])

        self._clean_df()

    def _prepare_matchday_stats(self):

        home_stats = self.matchday_df[['season', 'division', 'matchday', 'team', 'rank', 
                                                       'avg_GF', 'avg_GA', 'home_avg_GF', 'home_avg_GA', 
                                                       'avg_last_points','rank_n_last_seasons']]
        
        away_stats = self.matchday_df[['season', 'division', 'matchday', 'team', 'rank', 
                                                       'avg_GF', 'avg_GA', 'away_avg_GF', 'away_avg_GA', 
                                                       'avg_last_points','rank_n_last_seasons']]

        home_stats = home_stats.rename(columns={
            'team': 'home_team',
            'rank': 'home_rank',
            'avg_GF': 'home_total_avg_GF',
            'avg_GA': 'home_total_avg_GA',
            'home_avg_GF': 'home_GF_avg',
            'home_avg_GA': 'home_GA_avg',
            'avg_last_points': 'home_avg_points_last_5',
            'rank_n_last_seasons' : 'home_rank_5_last_seasons'
        })

        away_stats = away_stats.rename(columns={
            'team': 'away_team',
            'rank': 'away_rank',
            'avg_GF': 'away_total_avg_GF',
            'avg_GA': 'away_total_avg_GA',
            'away_avg_GF': 'away_GF_avg',
            'away_avg_GA': 'away_GA_avg',
            'avg_last_points': 'away_avg_points_last_5',
            'rank_n_last_seasons' : 'away_rank_5_last_seasons'

        })
        
        self._adjust_matchday(home_stats,away_stats)

    def _adjust_matchday(self,home_stats,away_stats):
        
        self.df["matchday"] = self.df["matchday"] - 1
        
        self.df = self.df.merge(home_stats, on=['season', 'division', 'matchday', 'home_team'], how='left')
        self.df = self.df.merge(away_stats, on=['season', 'division', 'matchday', 'away_team'], how='left')
        
        self.df["matchday"] = self.df["matchday"] + 1

    def _clean_df(self):
        
        self.df.drop(['date', 'time',
       'score', 'home_score', 'away_score', 'home_W',
       'home_L', 'home_T', 'away_W', 'away_L', 'away_T'],axis=1,
       inplace=True)
