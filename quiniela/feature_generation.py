import pandas as pd
import numpy as np


class FeatureGenerator:
    

    def apply_features(self, quiniela_df, list_features=[]):

        if len(list_features) == 0 or 'avg_goals' in list_features:
            self._avg_goals(quiniela_df)
        if len(list_features) == 0 or 'last_n_matchdays' in list_features:
            self._last_n_matchdays(quiniela_df)
        if len(list_features) == 0 or 'rank_n_last_seasons' in list_features:
            self._last_n_ranks(quiniela_df)
        if 'versus_features' in list_features:
            self._versus_features(quiniela_df)

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

        self._fill_missing_ranks(quiniela_df)


    def _fill_missing_ranks(self,quiniela_df):
        
        max_rank_by_season = quiniela_df.match_standings.groupby(['season', 'division'])['rank'].max().reset_index()
        max_rank_by_season.columns = ['season', 'division', 'max_rank']
        
        max_rank_div1_by_season = quiniela_df.match_standings[quiniela_df.match_standings['division'] == 1].groupby('season')['rank'].max().reset_index()
        max_rank_div1_by_season.columns = ['season', 'max_rank_div1']

        quiniela_df.match_standings = quiniela_df.match_standings.merge(max_rank_by_season, on=['season', 'division'], how='left')
        quiniela_df.match_standings = quiniela_df.match_standings.merge(max_rank_div1_by_season, on='season', how='left')

        quiniela_df.match_standings['rank_n_last_seasons'] = np.where(
            quiniela_df.match_standings['rank_n_last_seasons'].isna(),
            np.where(
                quiniela_df.match_standings['season'] == 1929, 
                0, 
                np.where(
                    quiniela_df.match_standings['division'] == 2,
                    quiniela_df.match_standings['max_rank_div1'] + quiniela_df.match_standings['max_rank'] + 1,
                    quiniela_df.match_standings['max_rank'] + 1  
                )
            ),
            quiniela_df.match_standings['rank_n_last_seasons']
        )
        
        quiniela_df.match_standings.drop(columns=['max_rank', 'max_rank_div1'],inplace=True)

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
    
    def _versus_features(self,matchday_df):
        
        matchday_df.df["dif_rank"] = matchday_df.df["home_rank"] - matchday_df.df["away_rank"]
        matchday_df.df["expected_home_goals"] = (matchday_df.df["home_GF_avg"] + matchday_df.df["away_GA_avg"]) / 2
        matchday_df.df["expected_away_goals"] = (matchday_df.df["away_GF_avg"] + matchday_df.df["home_GA_avg"]) / 2
        matchday_df.df["dif_last5_points"] = matchday_df.df["home_avg_points_last_5"] - matchday_df.df["away_avg_points_last_5"]
        matchday_df.df["avg_home_goals_total"] = (matchday_df.df["home_total_avg_GF"] + matchday_df.df["away_total_avg_GA"]) / 2
        matchday_df.df["avg_away_goals_total"] = (matchday_df.df["away_total_avg_GF"] + matchday_df.df["home_total_avg_GA"]) / 2
        matchday_df.df["dif_previous_ranks"] = matchday_df.df["home_rank_5_last_seasons"] - matchday_df.df["away_rank_5_last_seasons"]

        matchday_df.df.fillna(0)