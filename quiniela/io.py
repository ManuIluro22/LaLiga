import sqlite3
import pandas as pd
import settings
from quiniela.structure import LaLigaDataframe

def load_matchday(season, division, matchday):
    print(f"Loading matchday {matchday} in season {season}, division {division}...")

    actual_season = int(season[-4:])
    prev_10_season = int(season[-4:])-10
    seasons_tuple = tuple(f"{year}-{year + 1}" for year in range(prev_10_season, actual_season))    
    
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        data = pd.read_sql(f"""
                SELECT * FROM Matches
                    WHERE season IN {seasons_tuple}
            """, conn)
    if data.empty:
        raise ValueError("There is no matchday data for the values given")
    
    master = LaLigaDataframe(data.copy())


    master.generate_features()
    final = master.generate_matchday_dataframe()

    predict_data = final.df.loc[
       (final.df.season == actual_season) & 
       (final.df.division == division) &
       (final.df.matchday == matchday)]
    
    return predict_data



def load_historical_data(seasons):

    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        if seasons == "all":
            data = pd.read_sql("SELECT * FROM Matches", conn)
        else:
            data = pd.read_sql(f"""
                SELECT * FROM Matches
                    WHERE season IN {tuple(seasons)}
            """, conn)
    if data.empty:
        raise ValueError(f"No data for seasons {seasons}")
    return data


def save_predictions(predictions):

    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        predictions.to_sql(name="Predictions", con=conn, if_exists="append", index=False)
