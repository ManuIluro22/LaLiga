#!/usr/bin/env python
import logging
import argparse
from datetime import datetime
import pandas as pd
import settings
from quiniela import data_preprocessing, models, io
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from quiniela.structure import LaLigaDataframe

def parse_seasons(value):
    if value == "all":
        return "all"
    seasons = []
    for chunk in value.split(","):
        if ":" in chunk:
            try:
                start, end = map(int, chunk.split(":"))
                assert start < end
            except Exception:
                raise argparse.ArgumentTypeError(f"Unexpected format for seasons {value}")
            for i in range(start, end):
                seasons.append(f"{i}-{i+1}")
        else:
            try:
                start, end = map(int, chunk.split("-"))
                assert start == end - 1
            except Exception:
                raise argparse.ArgumentTypeError(f"Unexpected format for seasons {value}")
            seasons.append(chunk)
    return seasons


parser = argparse.ArgumentParser()
task_subparser = parser.add_subparsers(help='Task to perform', dest='task')
train_parser = task_subparser.add_parser("train")
train_parser.add_argument(
    "--training_seasons",
    default="all",
    type=parse_seasons,
    help="Seasons to use for training. Write them separated with ',' or use range with ':'. "
         "For instance, '2004:2006' is the same as '2004-2005,2005-2006'. "
         "Use 'all' to train with all seasons available in database.",
)
train_parser.add_argument(
    "--model_name",
    default="my_quiniela.model",
    help="The name to save the model with.",
)
predict_parser = task_subparser.add_parser("predict")
predict_parser.add_argument(
    "season",
    help="Season to predict",
)
predict_parser.add_argument(
    "division",
    type=int,
    choices=[1, 2],
    help="Division to predict (either 1 or 2)",
)
predict_parser.add_argument(
    "matchday",
    type=int,
    help="Matchday to predict",
)
predict_parser.add_argument(
    "--model_name",
    default="my_quiniela.model",
    help="The name of the model you want to use.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(
        filename=settings.LOGS_PATH / f"{args.task}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
    )
    if args.task == "train":
        logging.info(f"Training LaQuiniela model with seasons {args.training_seasons}")
        model = models.QuinielaModel()
        training_data = io.load_historical_data(args.training_seasons)
        model.train(training_data)
        model.save(settings.MODELS_PATH / args.model_name)
        print(f"Model succesfully trained and saved in {settings.MODELS_PATH / args.model_name}")
    if args.task == "predict":
        logging.info(f"Predicting matchday {args.matchday} in season {args.season}, division {args.division}")
        #loaded_model = models.QuinielaModel.load(settings.MODELS_PATH / args.model_name)
        
        predict_data = io.load_matchday(args.season, args.division, args.matchday)
        print("\n Data successfully loaded and processed.\n")
        
        training_data = io.load_historical_data("all")
        master = LaLigaDataframe(training_data.copy())
        master.generate_features()
        final = master.generate_matchday_dataframe()
        X_train = final.df.drop(columns=['result'])
        y_train = final.df['result']

        home_encoder = LabelEncoder()
        away_encoder = LabelEncoder()

        predict_data['home_team_orig'] = predict_data['home_team']
        predict_data['away_team_orig'] = predict_data['away_team']

        predict_data['home_team_orig'] = predict_data['home_team']
        predict_data['away_team_orig'] = predict_data['away_team']

        # Separate training data
        training_data = final.df.copy()

        # Convert 'X' to 0 in the result column
        training_data['result'] = training_data['result'].replace('X', 0)
        training_data['result'] = training_data['result'].astype(int)

        # Prepare features for training
        X_train = training_data.drop(columns=['result'])
        y_train = training_data['result']

        # Encode team names
        X_train['home_team'] = home_encoder.fit_transform(X_train['home_team'])
        X_train['away_team'] = away_encoder.fit_transform(X_train['away_team'])
        X_train = pd.get_dummies(X_train, columns=['season'])

        # Prepare prediction data
        predict_data['home_team'] = home_encoder.transform(predict_data['home_team'])
        predict_data['away_team'] = away_encoder.transform(predict_data['away_team'])
        X_pred = predict_data

        # Ensure prediction data has same columns as training data
        missing_cols = set(X_train.columns) - set(X_pred.columns)
        for col in missing_cols:
            X_pred[col] = 0
        X_pred = X_pred[X_train.columns]

        # Train and predict
        model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        predict_data['pred'] = model.predict(X_pred)

        # Display results
        print("=" * 70)
        print("Predicted results: ")
        print("(1 = home win, 2 = away win, X = tie)")
        print("=" * 70)

        for _, row in predict_data.iterrows():
            result = "1" if row['pred'] == 1 else "2" if row['pred'] == 2 else "X"
            print(f"{row['home_team_orig']:^30s} vs {row['away_team_orig']:^30s} --> {result}")
        print("=" * 70)
        
