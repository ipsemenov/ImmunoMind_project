import logging
import argparse
import sys
from pathlib import Path
import warnings

import pandas as pd

from calculations import (SetLogger,
                          train,
                          predict,
                          benchmarking)
warnings.simplefilter('ignore')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='scRNA-seq classifiers', epilog='Enjoy the program! :)',
                                     formatter_class=argparse.RawTextHelpFormatter)
    sub_parsers = parser.add_subparsers(dest='subparser')

    # train subparser
    parser_train = sub_parsers.add_parser('train', help='Create new classifier and train it on passed dataset(s)')
    parser_train.add_argument('-i', '--input', type=str, help='Path to scRNA-seq dataset(s)',
                              required=True, metavar='')
    parser_train.add_argument('-o', '--output', type=str, help='Path to the folder with results',
                              required=True, metavar='')
    parser_train.add_argument('-p', '--path', type=str, help='Path to the folder with models',
                              required=True, metavar='')
    parser_train.add_argument('--models', type=str, nargs='+',
                              help='Names of the models to train. Available: svc, rforest, lgbm, corr',
                              required=True, metavar='')
    parser_train.add_argument('--n_genes', type=int, help='Number of genes to select for models training',
                              required=True, metavar='')
    parser_train.add_argument('--threshold', type=int, help='1 - if we want to use adaptive threshold, else - 0',
                              required=True, metavar='')
    parser_train.add_argument('--n_jobs', type=int, help='Number of jobs to run in parallel.',
                              required=True, metavar='')

    # predict subparser
    parser_predict = sub_parsers.add_parser('predict',
                                            help='Process new dataset with prepared models and predict cell types')
    parser_predict.add_argument('-i', '--input', type=str, help='Path to scRNA-seq dataset',
                                required=True, metavar='')
    parser_predict.add_argument('-o', '--output', type=str, help='Path to the folder with results',
                                required=True, metavar='')
    parser_predict.add_argument('-p', '--path', type=str, help='Path to the folder with models',
                                required=True, metavar='')
    parser_predict.add_argument('--models', type=str, nargs='+',
                                help='Names of the models to train. Available: svc, rforest, lgbm, corr',
                                required=True, metavar='')
    parser_predict.add_argument('--n_genes', type=int, help='Number of genes to select for models training',
                                required=True, metavar='')
    parser_predict.add_argument('--benchmarking', type=int, help='0 if we want to run benchmarking, else 1',
                                required=True, metavar='')
    parser_predict.add_argument('--n_jobs', type=int, help='Number of jobs to run in parallel.',
                                required=True, metavar='')
    args = parser.parse_args()

    # check whether output folder exists
    if not Path(args.output).exists():
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # create argparse logger
    SetLogger(logger_name='argparse', output_path=args.output)
    logger = logging.getLogger('argparse')

    # check whether input file exists
    if not Path(args.input).exists():
        logger.warning('File {} does not exist!'.format(args.input))
        logger.info('Abort calculations. Specify correct path to the file.')
        sys.exit()

    # getting data
    logger.info('Start reading file...')
    df = pd.read_csv(args.input)
    logger.info('End reading file.')
    file_name = str(Path(args.input).stem)

    # models evaluation
    if args.subparser == 'train':
        train(df=df, logger_name='argparse', file_name=file_name, path_to_models=args.path, models=args.models,
              n_genes=args.n_genes, threshold=args.threshold, output_path=args.output, n_jobs=args.n_jobs)

    elif args.subparser == 'predict':
        if args.benchmarking:
            benchmarking(df=df, logger_name='argparse', file_name=file_name, path_to_models=args.path,
                         models=args.models, n_genes=args.n_genes, output_path=args.output)
        else:
            predict(df=df, logger_name='argparse', file_name=file_name, path_to_models=args.path,
                    models=args.models, n_genes=args.n_genes, output_path=args.output)
