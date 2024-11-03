import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from quiniela.structure import LaLigaDataframe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE



class QuinielaModel:

    def train(self, train_data):
        master = LaLigaDataframe(train_data.copy())
        master.generate_features()
        final = master.generate_matchday_dataframe()

        y_train = final.df.loc[:, 'result']
        X_train = final.df.drop(columns=['season', 'division', 'home_team', 'away_team', 'result'])

        # Create a pipeline for scaling, PCA, and SMOTE
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.pca = PCA(n_components=0.85)
        X_pca = self.pca.fit_transform(X_scaled)

        smote = SMOTE(random_state=1)
        X_train, y_train = smote.fit_resample(X_pca, y_train)

        model = LogisticRegression(C=0.01, penalty='l1', solver='saga')
        model.fit(X_train, y_train)

        self.model = model
        return model

    def predict(self, predict_data):
        X = predict_data.drop(columns=['season', 'division', 'home_team', 'away_team', 'result'])
        X_scaled = self.scaler.transform(X)  
        X_pca = self.pca.transform(X_scaled)  
        
        predictions = self.model.predict(X_pca)
        probabilities = self.model.predict_proba(X_pca)
    
        max_probabilities = probabilities.max(axis=1)

        return predictions,max_probabilities

    @classmethod
    def load(cls, filename):
        """ Load model, scaler, and PCA from file """
        with open(filename, "rb") as f:
            model_instance = pickle.load(f)
            assert isinstance(model_instance, cls)  
        return model_instance

    def save(self, filename):
        """ Save the model, scaler, and PCA in a file """
        with open(filename, "wb") as f:
            pickle.dump(self, f)