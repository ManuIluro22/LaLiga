import pandas as pd

def clean_data(df):
    df = df.loc[~df['score'].isnull()].copy()

    # Separate home and away scores
    df[['home_score', 'away_score']] = df['score'].str.split(':', expand=True)
    df['home_score'] = pd.to_numeric(df['home_score'])
    df['away_score'] = pd.to_numeric(df['away_score'])

    # Define the result column (home win, away win, tie)
    df['result'] = df.apply(lambda row: '1' if row['home_score'] > row['away_score']
                                        else '2' if row['home_score'] < row['away_score']
                                        else 'X', axis=1)
    
    # Convert 'date' column to datetime with specified format and add weekday columns
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
    df['weekday'] = df['date'].dt.day_name()
    df['weekday_num'] = df['date'].dt.dayofweek
    
    # Drop 'weekday' if unnecessary
    df = df.drop('weekday', axis=1)

    return df

def calculate_gd_5_game(df):
    # Initialise columns
    df['gd_5_game_home'] = None
    df['gd_5_game_away'] = None
    
    teams = df['home_team'].unique()
    for team in teams:
        team_df = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        team_df['goal_difference'] = team_df.apply(
            lambda x: x['home_score'] - x['away_score'] if x['home_team'] == team else x['away_score'] - x['home_score'], axis=1
        )
        team_df['gd_5_game'] = team_df['goal_difference'].rolling(window=5).sum().shift(1)
        
        # Assign gd_5_game to the appropriate column based on home or away
        df.loc[team_df[team_df['home_team'] == team].index, 'gd_5_game_home'] = team_df.loc[team_df['home_team'] == team, 'gd_5_game']
        df.loc[team_df[team_df['away_team'] == team].index, 'gd_5_game_away'] = team_df.loc[team_df['away_team'] == team, 'gd_5_game']
    
    return df

def calculate_form_10_game(df):

    # Initialise rows
    df['form_10_game_home'] = None
    df['form_10_game_away'] = None

    teams = df['home_team'].unique()
    for team in teams:
        team_df = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        team_df['win'] = team_df.apply(
            lambda x: 1 if (x['home_team'] == team and x['home_score'] > x['away_score']) or 
                           (x['away_team'] == team and x['away_score'] > x['home_score']) else 0, axis=1
        )
        team_df['form_10_game'] = team_df['win'].rolling(window=10).sum().shift(1)
        
        # Assign form_10_game to the appropriate column based on home or away
        df.loc[team_df[team_df['home_team'] == team].index, 'form_10_game_home'] = team_df.loc[team_df['home_team'] == team, 'form_10_game']
        df.loc[team_df[team_df['away_team'] == team].index, 'form_10_game_away'] = team_df.loc[team_df['away_team'] == team, 'form_10_game']
    
    return df

def calculate_live_rank(df):
    
    # Initialise rows
    df['live_rank_home'] = None
    df['live_rank_away'] = None
    
    teams = df['home_team'].unique()
    points_table = {team: 0 for team in teams}
    matchdays = sorted(df['matchday'].unique())
    
    for matchday in matchdays:
        day_df = df[df['matchday'] == matchday]
        for _, row in day_df.iterrows():
            home_team, away_team = row['home_team'], row['away_team']
            home_score, away_score = row['home_score'], row['away_score']
            
            # Update points based on match result
            if home_score > away_score:
                points_table[home_team] += 3
            elif away_score > home_score:
                points_table[away_team] += 3
            else:
                points_table[home_team] += 1
                points_table[away_team] += 1
        
        # Sort teams by points and assign ranks
        sorted_teams = sorted(points_table.items(), key=lambda x: x[1], reverse=True)
        rank_dict = {team: rank+1 for rank, (team, _) in enumerate(sorted_teams)}
        
        # Update live rank for this matchday for both home and away teams
        for _, row in day_df.iterrows():
            df.loc[row.name, 'live_rank_home'] = rank_dict[row['home_team']]
            df.loc[row.name, 'live_rank_away'] = rank_dict[row['away_team']]
    
    return df

# --> I think we should split this into smaller chunks and then nest
# --  into a main feature

def calculate_goals_scored(df):
    home_goals = df.groupby('home_team')['home_score'].sum().reset_index()
    home_goals.columns = ['team', 'home_goals_team_home']
    
    away_goals = df.groupby('away_team')['away_score'].sum().reset_index()
    away_goals.columns = ['team', 'away_goals_team_away']

    # Merge home and away goals to get total goals
    all_time = pd.merge(home_goals, away_goals, on='team', how='outer').fillna(0)
    all_time['total_goals'] = all_time['home_goals_team_home'] + all_time['away_goals_team_away']

    df = df.merge(all_time[['team','home_goals_team_home','total_goals']],
                  left_on='home_team', right_on='team', how='outer').fillna(0)
    df = df.merge(all_time[['team','away_goals_team_away','total_goals']],
                  left_on='away_team', right_on='team', suffixes=('_home', '_away'), how='outer').fillna(0)
    df = df.drop(['team_home', 'team_away'], axis=1)
    
    return df


def calculate_goals_conceded(df):
    home_conceding = df.groupby('home_team')['away_score'].sum().reset_index()
    home_conceding.columns = ['team', 'home_concedes_team_home']
    away_conceding = df.groupby('away_team')['home_score'].sum().reset_index()
    away_conceding.columns = ['team', 'away_concedes_team_away']

    # Merge conceding data with total goals data
    all_time_concedes = pd.merge(home_conceding, away_conceding, on='team', how='outer').fillna(0)
    all_time_concedes['total_concedes'] = all_time_concedes['home_concedes_team_home'] + all_time_concedes['away_concedes_team_away']

    df = df.merge(all_time_concedes[['team','home_concedes_team_home','total_concedes']],
                  left_on='home_team', right_on='team', how='outer').fillna(0)
    df = df.merge(all_time_concedes[['team','away_concedes_team_away','total_concedes']],
                  left_on='away_team', right_on='team', suffixes=('_home', '_away'), how='outer').fillna(0)
    df = df.drop(['team_home', 'team_away'], axis=1)
    return df


def calculate_appearances(df):
    home_counts = df['home_team'].value_counts()
    away_counts = df['away_team'].value_counts()
    total_counts = home_counts.add(away_counts, fill_value=0)

    # Map appearances onto the DataFrame
    df['home_app_home'] = df['home_team'].map(home_counts)
    df['home_app_total'] = df['home_team'].map(total_counts)
    df['away_app_away'] = df['away_team'].map(away_counts)
    df['away_app_total'] = df['away_team'].map(total_counts)
    
    return df


def calculate_scor_conced_ratio(df):
    df['home_goals_ratio'] = df['home_goals_team_home'] / df['home_app_home']
    df['away_goals_ratio'] = df['away_goals_team_away'] / df['away_app_away']
    df['total_goals_home_ratio'] = df['total_goals_home'] / df['home_app_total']
    df['total_goals_away_ratio'] = df['total_goals_away'] / df['away_app_total']
    df['concedes_home_ratio'] = df['home_concedes_team_home'] / df['home_app_home']
    df['concedes_away_ratio'] = df['away_concedes_team_away'] / df['away_app_away']
    df['total_concedes_home_ratio'] = df['total_concedes_home'] / df['home_app_total']
    df['total_concedes_away_ratio'] = df['total_concedes_away'] / df['away_app_total']
    
    return df


def calculate_goal_diff(df):
    df['goal_diff_home'] = df['home_goals_team_home'] - df['home_concedes_team_home']
    df['goal_diff_home_total'] = df['total_goals_home'] - df['total_concedes_home']
    df['goal_diff_away'] = df['away_goals_team_away'] - df['away_concedes_team_away']
    df['goal_diff_away_total'] = df['total_goals_away'] - df['total_concedes_away']
    df['goal_diff_home_ratio'] = df['home_goals_ratio'] - df['concedes_home_ratio']
    df['goal_diff_home_total_ratio'] = df['total_goals_home_ratio'] - df['total_concedes_home_ratio']
    df['goal_diff_away_ratio'] = df['away_goals_ratio'] - df['concedes_away_ratio']
    df['goal_diff_away_total_ratio'] = df['total_goals_away_ratio'] - df['total_concedes_away_ratio']

    return df


def generate_features(df):
    # Include remaining features
    df = calculate_goals_scored(df)
    df = calculate_goals_conceded(df)
    df = calculate_appearances(df)
    df = calculate_scor_conced_ratio(df)
    df = calculate_goal_diff(df)
    df = calculate_form_10_game(df)
    df = calculate_gd_5_game(df)
    df = calculate_live_rank(df)

    return df

def get_X_y(df):

    y = df.loc[:,'result']
    cols_to_drop = ['score','home_score','away_score','result','date', 'time']
    X = df.drop(columns=cols_to_drop)
    X = pd.get_dummies(X, columns=['season','home_team','away_team'])
    
    return X, y