import pickle
from quiniela.data_preprocessing import clean_data, generate_features, get_X_y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

class QuinielaModel:

    def train(self, train_data):
        train_data = clean_data(train_data)
        train_data = generate_features(train_data)
        # Do something here to train the model
        # --> Do we need to include the test args here??
        """ Train a ML model from the train data """
        X, y = get_X_y(train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        model = HistGradientBoostingClassifier()
        model.fit(X_train, y_train)
        return model

    def predict(self, predict_data):
        predict_data = clean_data(predict_data)
        predict_data = generate_features(predict_data)
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
