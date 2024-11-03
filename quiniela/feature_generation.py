import pandas as pd
import numpy as np


class FeatureGenerator:
    """
    A class responsible for generating various features for match predictions
    and analysis in La Liga.

    This class provides methods to compute features based on match statistics
    and team performance over recent matchdays.
    The features generated can enhance the predictive capabilities of machine
    learning models by providing
    contextual information about the teams' past performances.

    Attributes:
        None
    """

    def apply_features(self, quiniela_df, selected_features=[]):
        """
        Applies specified feature generation functions to the given
        LaLigaDataframe instance.

        Args:
            quiniela_df (LaLigaDataframe): An instance containing match and
            standings data.
            selected_features (list, optional): A list of feature names to be
            generated (avg_goals, last_n_matchdays, rank_n_last_seasons and
            versus_features). If empty, all features will be generated.
        """
        if (len(selected_features) == 0 or
                'avg_goals' in selected_features):
            self._avg_goals(quiniela_df)
        if (len(selected_features) == 0 or
                'last_5_matchdays' in selected_features):
            self._last_5_matchdays(quiniela_df)
        if (len(selected_features) == 0 or
                'rank_5_last_seasons' in selected_features):
            self._last_5_ranks(quiniela_df)
        if 'versus_features' in selected_features:
            self._versus_features(quiniela_df)

    def _last_5_ranks(self, quiniela_df):
        """
        Computes the average ranking of each team over the last 5 seasons and
        merges it with the match standings.

        Args:
            quiniela_df (LaLigaDataframe): An instance containing match and
            standings data.
        """
        last_game_rankings = quiniela_df.match_standings.sort_values(by=[
            'season',
            'division',
            'team',
            'matchday']).groupby([
                'season',
                'division',
                'team']).last().reset_index()

        last_game_rankings.loc[
            last_game_rankings['division'] == 2, 'rank'] += 20

        last_game_rankings["rank_5_last_seasons"] = last_game_rankings.groupby(
            'team')['rank'].transform(lambda x: x.rolling(window=5,
                                                          min_periods=1)
                                                 .mean())

        last_game_rankings["season"] = last_game_rankings["season"] + 1

        quiniela_df.match_standings = quiniela_df.match_standings.merge(
            last_game_rankings[[
                'season',
                'division',
                'team',
                'rank_5_last_seasons']],
            on=['season', 'division', 'team'],
            how='left'
        )
        self._fill_missing_ranks(quiniela_df)

    def _fill_missing_ranks(self, quiniela_df):
        """
        Fills in missing rank values based on the maximum rank
        (last rank of second division + 1).

        Args:
            quiniela_df (LaLigaDataframe): An instance containing
            match and standings data.
        """
        max_rank_by_season = (quiniela_df.match_standings
                              .groupby(['season', 'division'])['rank']
                              .max().reset_index())
        max_rank_by_season.columns = ['season', 'division', 'max_rank']

        max_rank_div1_by_season = (quiniela_df
                                   .match_standings[
                                       quiniela_df
                                       .match_standings['division'] == 1]
                                   .groupby('season')['rank']
                                   .max().reset_index())
        max_rank_div1_by_season.columns = ['season', 'max_rank_div1']

        quiniela_df.match_standings = (quiniela_df
                                       .match_standings
                                       .merge(max_rank_by_season,
                                              on=['season', 'division'],
                                              how='left'))
        quiniela_df.match_standings = (quiniela_df
                                       .match_standings
                                       .merge(max_rank_div1_by_season,
                                              on='season', how='left'))

        quiniela_df.match_standings['rank_5_last_seasons'] = np.where(
            quiniela_df.match_standings['rank_5_last_seasons'].isna(),
            np.where(
                quiniela_df.match_standings['season'] == 1929,
                0,
                np.where(
                    quiniela_df.match_standings['division'] == 2,
                    (quiniela_df.match_standings['max_rank_div1'] +
                        quiniela_df.match_standings['max_rank'] + 1),
                    quiniela_df.match_standings['max_rank'] + 1
                )
            ),
            quiniela_df.match_standings['rank_5_last_seasons']
        )

        quiniela_df.match_standings.drop(
            columns=['max_rank', 'max_rank_div1'], inplace=True)

    def _avg_goals(self, quiniela_df):
        """
        Calculates average goals scored and conceded for home and away teams.

        Args:
            quiniela_df (LaLigaDataframe): An instance containing
            match and standings data.
        """
        quiniela_df.match_standings['home_avg_GF'] = (
            quiniela_df.match_standings['home_GF'] /
            quiniela_df.match_standings['matchday']
        )
        quiniela_df.match_standings['home_avg_GA'] = (
            quiniela_df.match_standings['home_GA'] /
            quiniela_df.match_standings['matchday']
        )
        quiniela_df.match_standings['away_avg_GF'] = (
            quiniela_df.match_standings['away_GF'] /
            quiniela_df.match_standings['matchday']
        )
        quiniela_df.match_standings['away_avg_GA'] = (
            quiniela_df.match_standings['away_GA'] /
            quiniela_df.match_standings['matchday']
        )
        quiniela_df.match_standings['avg_GF'] = (
            quiniela_df.match_standings['GF'] /
            quiniela_df.match_standings['matchday']
        )
        quiniela_df.match_standings['avg_GA'] = (
            quiniela_df.match_standings['GA'] /
            quiniela_df.match_standings['matchday']
        )

    def _last_5_matchdays(self, quiniela_df):
        """
        Generates results from the last 5 matchdays for each team
        and calculates average points.

        Args:
            quiniela_df (LaLigaDataframe): An instance containing match
            and standings data.
        """
        quiniela_df.team_results['result'] = (
            quiniela_df.team_results.apply(
                lambda row: 'W'
                if row['W'] == 1
                else 'L'
                if row['L'] == 1
                else 'T',
                axis=1)
        )
        quiniela_df.team_results.sort_values(['season',
                                              'division',
                                              'team',
                                              'matchday'], inplace=True)

        last_results = (
            quiniela_df
            .team_results
            .groupby(['season',
                      'division',
                      'team'])['result']
            .apply(lambda x: np.array(
                [np.array(rolling_list) for rolling_list in x.rolling(5)], 
                dtype=object)))

        last_results = (
            np.concatenate(
                last_results.values)
              .reshape(-1))

        quiniela_df.match_standings.sort_values(['season', 'division', 'team', 'matchday'], inplace=True)
        quiniela_df.match_standings.reset_index(inplace=True)
        quiniela_df.match_standings["avg_5_last_points"] = (pd.Series(last_results).apply(self._calculate_avg_points))

    def _calculate_avg_points(self, last_n):
        """
        Calculates the average points scored from the last n match results.

        Args:
            last_n (ndarray): A list of results from the last n matchdays.

        Returns:
            float: The average points scored from the results.
        """
        points = {'W': 3, 'T': 1, 'L': 0}
        total_points = np.mean([points[res] for res in last_n if res in points])
        return total_points if len(last_n) > 0 else 0
    
    def _versus_features(self,matchday_df):
        """
        Generates comparative features between home and away teams for each matchday.

        Args:
            matchday_df (MatchdayDataframe): An instance containing matchday-specific data.
        """        
        matchday_df.df["dif_rank"] = (matchday_df.df["home_rank"] - 
                                      matchday_df.df["away_rank"])
        matchday_df.df["expected_home_goals"] = ((matchday_df
                                                  .df["home_GF_avg"] + 
                                                  matchday_df.df["away_GA_avg"]) / 2)
        matchday_df.df["expected_away_goals"] = ((matchday_df
                                                  .df["away_GF_avg"] + 
                                                  matchday_df.df["home_GA_avg"]) / 2)
        matchday_df.df["dif_last_5_points"] = (matchday_df
                                               .df[f"home_avg_points_last_5"] - 
                                               matchday_df.df[f"away_avg_points_last_5"])
        matchday_df.df["avg_home_goals_total"] = ((matchday_df
                                                   .df["home_total_avg_GF"] + 
                                                   matchday_df
                                                   .df["away_total_avg_GA"]) / 2)
        matchday_df.df["avg_away_goals_total"] = ((matchday_df
                                                   .df["away_total_avg_GF"] + 
                                                   matchday_df
                                                   .df["home_total_avg_GA"]) / 2)
        matchday_df.df["dif_previous_5_ranks"] = (matchday_df
                                                  .df[f"home_rank_5_last_seasons"] - 
                                                  matchday_df.df[f"away_rank_5_last_seasons"])
        
        matchday_df.df = (matchday_df
                          .df.groupby(['home_team',
                                       'away_team'],
                                    group_keys=False).apply(
                                    self._calculate_H2H_stats))

        matchday_df.df.fillna(0, inplace=True)

    def _calculate_H2H_stats(self, matches):
        """
        Calculates head-to-head (H2H) statistics for the last n matches between two teams, 
        based on historical match data. The statistics generated include the number of wins, ties, 
        and goals for both home and away teams over the previous 10 encounters.

        Args:
        matches (DataFrame): A Pandas DataFrame containing historical match data. 

        Returns:
        matches (DataFrame): The original DataFrame with additional columns for H2H statistics:
        """
        matches = matches.sort_values('season')

        matches['H2H_wins_home_last_10'] = (matches['home_team'] == matches['home_team'].iloc[0]) * matches['home_W'].shift().rolling(window=10, min_periods=0).sum()
        matches['H2H_wins_away_last_10'] = (matches['away_team'] == matches['away_team'].iloc[0]) * matches['home_L'].shift().rolling(window=10, min_periods=0).sum()
        matches['H2H_ties_last_10'] = matches['home_T'].shift().rolling(window=10, min_periods=0).sum()
        matches['H2H_goals_home_last_10'] = (matches['home_team'] == matches['home_team'].iloc[0]) * matches['home_score'].shift().rolling(window=10, min_periods=0).sum()
        matches['H2H_goals_away_last_10'] = (matches['away_team'] == matches['away_team'].iloc[0]) * matches['away_score'].shift().rolling(window=10, min_periods=0).sum()

        return matches
