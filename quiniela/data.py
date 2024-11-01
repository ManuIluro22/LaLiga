import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

class QuinielaDataframe:

    def __init__(self, dataframe):

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

    def generate_features(self, list_features=None):

        feature_generator = FeatureGenerator()
        feature_generator.apply_features(self, list_features)

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

class FeatureGenerator:
    

    def apply_features(self, quiniela_df, list_features=None):

        if list_features is None or 'avg_goals' in list_features:
            self._avg_goals(quiniela_df)
        if list_features is None or 'last_n_matchdays' in list_features:
            self._last_n_matchdays(quiniela_df)
        if list_features is None or 'rank_n_last_seasons' in list_features:
            self._last_n_ranks(quiniela_df)

    def _last_n_ranks(self,quiniela_df,n=5):

        last_game_rankings = quiniela_df.match_standings.sort_values(by=['season', 'division', 'team', 'matchday']).groupby(['season', 'division', 'team']).last().reset_index()
        last_game_rankings.loc[last_game_rankings['division'] == 2, 'rank'] += 20
        last_game_rankings["rank_n_last_seasons"] = last_game_rankings.groupby('team')['rank'].transform(lambda x: x.rolling(window=5,min_periods=1).mean())
        last_game_rankings["season"] = last_game_rankings["season"] + 1
        quiniela_df.match_standings = quiniela_df.match_standings.merge(
            last_game_rankings[['season', 'division', 'team', 'rank_n_last_seasons']],
            on=['season', 'division', 'team'],
            how='left'
        )

        self._fill_missing_ranks(quiniela_df.match_standings)


    def _fill_missing_ranks(self,match_standings):
        
        
        max_rank_by_season = match_standings.groupby(['season', 'division'])['rank'].max().reset_index()
        max_rank_by_season.columns = ['season', 'division', 'max_rank']
        
        max_rank_div1_by_season = match_standings[match_standings['division'] == 1].groupby('season')['rank'].max().reset_index()
        max_rank_div1_by_season.columns = ['season', 'max_rank_div1']

        match_standings = match_standings.merge(max_rank_by_season, on=['season', 'division'], how='left')
        match_standings = match_standings.merge(max_rank_div1_by_season, on='season', how='left')

        match_standings['rank_n_last_seasons'] = np.where(
            match_standings['rank_n_last_seasons'].isna(),
            np.where(
                match_standings['season'] == 1929, 
                0, 
                np.where(
                    match_standings['division'] == 2,
                    match_standings['max_rank_div1'] + match_standings['max_rank'] + 1,
                    match_standings['max_rank'] + 1  
                )
            ),
            match_standings['rank_n_last_seasons']
        )
        
        match_standings.drop(columns=['max_rank', 'max_rank_div1'],inplace=True)
        print(match_standings.sort_values(by=['season', 'division', 'matchday', 'Pts', 'GD', 'GF'],
                                                                ascending=[True, True, True, False, False, False]))

    def _avg_goals(self, quiniela_df):

        quiniela_df.match_standings['home_avg_GF'] = quiniela_df.match_standings['home_GF'] / quiniela_df.match_standings['matchday']
        quiniela_df.match_standings['home_avg_GA'] = quiniela_df.match_standings['home_GA'] / quiniela_df.match_standings['matchday']
        quiniela_df.match_standings['away_avg_GF'] = quiniela_df.match_standings['away_GF'] / quiniela_df.match_standings['matchday']
        quiniela_df.match_standings['away_avg_GA'] = quiniela_df.match_standings['away_GA'] / quiniela_df.match_standings['matchday']
        quiniela_df.match_standings['avg_GF'] = quiniela_df.match_standings['GF'] / quiniela_df.match_standings['matchday']
        quiniela_df.match_standings['avg_GA'] = quiniela_df.match_standings['GA'] / quiniela_df.match_standings['matchday']

    def _last_n_matchdays(self, quiniela_df, n=5):

        quiniela_df.team_results['result'] = quiniela_df.team_results.apply(
            lambda row: 'W' if row['W'] == 1 else 'L' if row['L'] == 1 else 'T', axis=1
        )
        quiniela_df.team_results.sort_values(['season', 'division', 'team', 'matchday'], inplace=True)
        
        last_results = quiniela_df.team_results.groupby(['season', 'division', 'team'])['result'].apply(
            lambda x: np.array([np.array(rolling_list) for rolling_list in x.rolling(n)], dtype=object)
        )
        last_results = np.concatenate(last_results.values).reshape(-1)
        
        quiniela_df.match_standings.sort_values(['season', 'division', 'team', 'matchday'], inplace=True)
        quiniela_df.match_standings.reset_index(inplace=True)
        quiniela_df.match_standings["avg_last_points"] = pd.Series(last_results).apply(self._calculate_avg_points)

    def _calculate_avg_points(self, last_n):

        points = {'W': 3, 'T': 1, 'L': 0}
        total_points = np.mean([points[res] for res in last_n if res in points])
        return total_points if len(last_n) > 0 else 0