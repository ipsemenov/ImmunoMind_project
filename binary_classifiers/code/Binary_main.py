#!/usr/bin/env/python3
import pandas as pd
import argparse
import os

import Binary_classifiers as Bc

parser = argparse.ArgumentParser(description='Binary classifiers based on Xgboost')
subparsers = parser.add_subparsers()

parser_models = subparsers.add_parser('models', help='subcommand shows availible models located in ../models')
parser_models.add_argument("subcommand", type=str, default="models", nargs="?")

parser_predict = subparsers.add_parser('predict', help="subcommand predicts celltypes for given dataset")
parser_predict.add_argument("subcommand", type=str, default="predict", nargs="?")
parser_predict.add_argument("-i", "--input", type=str, help="specify path to dataset", metavar="")
parser_predict.add_argument("-m", "--model", type=str, help="specify model name without extenction (.zip), availible "
                                                            "models can be shown by models subcommand", metavar="")
parser_predict.add_argument("--adaptive_threshold", type=int, default=1, help="use adaptive threshold (1 - default) "
                                                                              "or not (0)", metavar="")

parser_benchmark = subparsers.add_parser('benchmark', help="subcommand returns metrics of prediction for this dataset")
parser_benchmark.add_argument("subcommand", type=str, default="benchmark", nargs="?")
parser_benchmark.add_argument("-i", "--input", type=str, help="specify path to dataset", metavar="")
parser_benchmark.add_argument("-m", "--model", type=str, help="specify model name without extenction (.zip), availible "
                                                              "models can be shown by models subcommand", metavar="")
parser_benchmark.add_argument("--adaptive_threshold", type=int, default=1, help="use adaptive threshold (1 - default) "
                                                                                "or not (0)", metavar="")
args = parser.parse_args()

if args.subcommand == "models":
    models = os.listdir("../models")
    for model in models:
        print(model[:-4])

elif args.subcommand == "predict":
    if args.input is None or args.model is None:
        raise AttributeError("You have not specified input dataset and model")
    dataset = pd.read_csv(args.input, index_col="cells")
    clf = Bc.BinaryClassifiersMaker()
    clf.load_model(args.model)
    if clf.check_prediction_availibility(dataset) is False:
        raise ValueError("Some classifiers of specified model are based on genes that are not represented in this "
                         "dataset. Cannot perform prediction")
    prediction = clf.predict(dataset, mode="dataframe", adaptive_threshold=bool(args.adaptive_threshold))
    print(prediction.to_string())

elif args.subcommand == "benchmark":
    if args.input is None or args.model is None:
        raise AttributeError("You have not specified input dataset and model")
    dataset = pd.read_csv(args.input, index_col="cells")
    clf = Bc.BinaryClassifiersMaker()
    clf.load_model(args.model)
    if clf.check_prediction_availibility(dataset) is False:
        raise ValueError("Some classifiers of specified model are based on genes that are not represented in this "
                         "dataset. Cannot perform prediction")
    clf.benchmark(args.input, mode="file", adaptive_threshold=bool(args.adaptive_threshold))
