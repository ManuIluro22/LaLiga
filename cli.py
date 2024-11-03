#!/usr/bin/env python
import logging
import argparse
from datetime import datetime
import pandas as pd
import settings
from quiniela import models, io
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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
        
        try:
            loaded_model = models.QuinielaModel.load(settings.MODELS_PATH / args.model_name)
        except FileNotFoundError:
            raise ValueError(f"The module with {args.model_name} name is not found")

        predict_data = io.load_matchday(args.season, args.division, args.matchday)
        
        prediction, probability = loaded_model.predict(predict_data)

        predict_data["pred"] = prediction
        predict_data["confidence"] = probability

        print(f"Matchday {args.matchday} - LaLiga - Division {args.division} - Season {args.season}")
        print("=" * 95)
        for _, row in predict_data.iterrows():
            print(f"{row['home_team']:^30s} vs {row['away_team']:^30s} --> {row['pred']} \
                  --- confidence: {row['confidence']*100:.2f}%")

        data_to_save = predict_data[["season",
                                     "division",
                                     "matchday",
                                     "home_team",
                                     "away_team",
                                     "pred"]]
        io.save_predictions(data_to_save)
