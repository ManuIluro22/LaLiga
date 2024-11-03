import pickle
from sklearn.ensemble import GradientBoostingClassifier
from quiniela.structure import LaLigaDataframe
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE



class QuinielaModel:
    """
    A class responsible for handling the training and prediction for Quiniela
    outcomes.

    This model uses a Gradient Boosting Classifier, and it scales and balance
    the data with SMOTE.
    It provides methods to train, predict, and save/load model instances.

    Attributes:
        model (GradientBoostingClassifier): Model used for training and
        prediction.
        scaler (StandardScaler): StandardScaler instance for feature scaling.
    """

    def __init__(self):
        """
        Initialize the model to a Gradient Boosting Classifier with the best
        found parameters and the scaler to a StandarScaler.

        """
        self.model = GradientBoostingClassifier(learning_rate=0.01,
                                                n_estimators=100)
        self.scaler = StandardScaler()

    def train(self, train_data):
        """
        Train the GradientBoosting model on the training data.

        Args:
            train_data (pd.DataFrame): Training data containing features
            and target.

        Returns:
            model: The trained model.
        """
        X_train, y_train = self._pipeline(train_data)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, predict_data):
        """
        Predict outcomes using the trained model and return probabilities.

        Args:
            predict_data (pd.DataFrame): Data containing features for
            prediction.

        Returns:
            tuple: Predicted labels and their maximum probabilities.
        """
        X = self._prepare_features(predict_data)

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        max_probabilities = probabilities.max(axis=1)
        return predictions, max_probabilities

    def _pipeline(self, train_data):

        """
        Prepare training data by generating features, scaling, and balancing.

        Args:
            train_data (pd.DataFrame): Raw training data.

        Returns:
            tuple: Scaled and balanced feature and target sets.
        """
        X, y = self._extract_features_and_target(train_data)
        X_scaled = self._scale_data(X)
        return self._balance_data(X_scaled, y)

    def _extract_features_and_target(self, data):
        """
        Extract features and target from the given data.

        Args:
            data (pd.DataFrame): Data containing features and target.

        Returns:
            tuple: Feature set X and target y.
        """
        master = LaLigaDataframe(data.copy())
        master.generate_features()
        final_df = master.generate_matchday_dataframe()
        y = final_df.df['result']
        X = final_df.df.drop(columns=['season', 'division', 'home_team',
                                      'away_team', 'result'])
        return X, y

    def _scale_data(self, X, train=True):
        """
        Scale features using the StandardScaler. Fits the scaler in the
        training process

        Args:
            X (pd.DataFrame): Data to be scaled.
            train (boolean, optional): Flag to fit or not the scaler. By
            default is set to True.

        Returns:
            pd.DataFrame: Scaled data.
        """
        return self.scaler.fit_transform(X) if (train) else \
            self.scaler.transform(X)

    def _balance_data(self, X, y):

        """
        Balance the dataset using SMOTE.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target set.

        Returns:
            tuple: Resampled features and target.
        """
        smote = SMOTE(random_state=1)
        return smote.fit_resample(X, y)

    def _prepare_features(self, data):
        """
        Extract and scale features for prediction.

        Args:
            data (pd.DataFrame): Data for which predictions are to be made.

        Returns:
            pd.DataFrame: Scaled features.
        """
        X = data.drop(columns=['season', 'division', 'home_team', 'away_team',
                               'result'])
        return self._scale_data(X, train=False)

    @classmethod
    def load(cls, filename):
        """Load the entire model instance, including scaler and classifier."""
        with open(filename, "rb") as f:
            model_instance = pickle.load(f)
            assert isinstance(model_instance, cls)
        return model_instance

    def save(self, filename):
        """Save the entire model instance to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)