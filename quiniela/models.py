import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from quiniela.feature_generation import FeatureGenerator
from quiniela.structure import LaLigaDataframe, MatchdayDataframe

class QuinielaModel:

    def train(self, train_data):
        master = LaLigaDataframe(train_data.copy())
        master.generate_features()
        final = master.generate_matchday_dataframe()
        print("=" * 70)
        print(final.df.head())

        y_train = master.df.loc[:,'result']
        X_train = master.df.drop(columns=['score','home_score','away_score','result','date', 'time'])
        X_train = pd.get_dummies(X_train, columns=['season','home_team','away_team'])
        
        model = HistGradientBoostingClassifier()
        model.fit(X_train, y_train)
        
        return model

    def predict(self, predict_data):
        
        predict_string = predict_data[
        (predict_data['result'] != 0) & 
        (predict_data['result'] != '0') | 
        (predict_data['result'] == 'X')
    ]['result'].astype(str).tolist()
         
        return predict_string

    @classmethod
    def load(cls, filename):
        """ Load model from file """
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) == cls
        return model

    def save(self, filename):
        """ Save a model in a file """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
