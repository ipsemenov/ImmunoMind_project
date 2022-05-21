import logging
import random
import time
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from pandas.plotting import table
from sklearn.model_selection import (RandomizedSearchCV,
                                     StratifiedKFold,
                                     train_test_split)
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

sys.path.append('./sclearn/')
try:
    from plotting import (visualize_cell_types,
                          visualize_residuals)
    from classifiers import CalibratedThresholdsClassifier
    from classifiers import CorrelationsClassifier
except BaseException:
    raise

PARAMS_RF = {'n_estimators': np.arange(1, 1010, 10),
             'max_depth': np.arange(1, 55, 5),
             'min_samples_split': np.arange(2, 25, 5)}

PARAMS_SVC = {'kernel': ['rbf', 'sigmoid', 'linear'],
              'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
              'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 200, 500, 700, 800, 1000]}

PARAMS_BOOSTING = {'n_estimators': np.arange(1, 1010, 10),
                   'max_depth':  np.arange(1, 55, 5),
                   'learning_rate': np.logspace(0.001, 10, 15),
                   'min_child_weight': np.arange(1, 5)}


def SetLogger(logger_name, output_path):
    '''
    Create custom logger and set its configuration

    :param logger_name: str, name of created logger
    :param output_path: str, path to file with logs
    '''

    logger = logging.getLogger(logger_name)  # Create a custom logger
    logger.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()  # Create handlers
    c_handler.setLevel(logging.INFO)
    f_handler = logging.FileHandler(str(Path(output_path, 'log.txt')))  # check whether output folder exists
    if not Path(output_path).exists():
        Path(output_path).mkdir(parent=True, exist_ok=True)
    f_handler.setLevel(logging.INFO)

    handler_format = logging.Formatter('%(levelname)s: %(message)s')  # Create formatters and add them to handlers
    c_handler.setFormatter(handler_format)
    f_handler.setFormatter(handler_format)

    logger.addHandler(c_handler)  # Add handlers to the logger
    logger.addHandler(f_handler)


def timedelta_to_hms(duration):
    '''
    Convert time of code execution from seconds to hms format
    :param duration: float, duration of code execution in seconds
    '''

    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = (duration % 3600) % 60

    return f'{hours}:{minutes}:{seconds}'


def rank_genes_train_set(X):
    '''
    Rank genes based on residuals of the model mean_exp ~ dropout_rate

    :param X: dataframe with gene expressions
    :return: residuals - array with residuals for further plotting
             sorted_genes - list with genes sorted by their importance
    '''

    dropout = (X == 0).sum(axis=0)
    dropout = (dropout / X.shape[0]) * 100
    mean = X.mean(axis=0)

    notzero = np.where((np.array(mean) > 0) & (np.array(dropout) > 0))[0]
    zero = np.where(~((np.array(mean) > 0) & (np.array(dropout) > 0)))[0]
    train_notzero = X.iloc[:, notzero]
    train_zero = X.iloc[:, zero]
    zero_genes = train_zero.columns

    dropout = dropout.iloc[notzero]
    mean = mean.iloc[notzero]

    dropout = np.array(dropout).reshape(-1, 1)
    mean = np.array(mean).reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(mean, dropout)

    residuals = dropout - reg.predict(mean)
    residuals = pd.Series(np.array(residuals).ravel(), index=train_notzero.columns)
    residuals = residuals.sort_values(ascending=False)
    sorted_genes = residuals.index
    sorted_genes = list(sorted_genes.append(zero_genes))

    return residuals, sorted_genes


def dataset_processing(X, file_name, path_to_models, model, n_genes, output_path, selected_genes=False):
    '''
    Filter the most important genes

    :param X: dataframe with gene expressions
    :param file_name: str, name of the file with scRNA-seq data (without extension)
    :param path_to_models: str, path to the folder with models
    :param model: str, name of the trained model
    :param n_genes: int, number of the most important genes to take into account
    :param output_path: str, path to the folder with results
    :param selected_genes: bool, 1 if file with selected genes exists
    :return: X_filtered - dataframe with filtered genes
    '''

    if not selected_genes:
        # genes selection
        residuals, sorted_genes = rank_genes_train_set(X=X)
        selected_genes = sorted_genes[:n_genes]
        random.shuffle(selected_genes)
        X_filtered = X[selected_genes]

        # save genes
        df_genes = pd.DataFrame(selected_genes, columns=['genes'])
        df_genes.to_csv(str(Path(path_to_models, f'{model}_{n_genes}_metadata.csv')), index=False)

        # visualize
        visualize_residuals(residuals=residuals,
                            n_genes=n_genes,
                            file_name=file_name,
                            output_path=output_path)

    else:
        df_genes = pd.read_csv(str(Path(path_to_models, f'{model}_{n_genes}_metadata.csv')))
        selected_genes = df_genes['genes'].tolist()
        X_filtered = X[selected_genes]

    return X_filtered


def save_metrics(metrics_dict, output_path, file_name):
    '''
    Save metrics (accuracy, precision, recall, f1_score) from models evaluation

    :param metrics_dict: dictionary with metrics
    :param output_path: str, path to the folder with results (dataframe and .png)
    :param file_name: str, name of the file with scRNA-seq data (without extension)
    '''

    # results in dataframe
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics = df_metrics.round(4)
    df_metrics.to_csv(str(Path(output_path, file_name + '_metrics.tsv')), sep='\t')

    # results in picture
    ax = plt.subplot(111, frame_on=False)  # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)
    table(ax, df_metrics, loc="center")
    plt.savefig(str(Path(output_path, file_name + '_metrics.png')), bbox_inches="tight")


def train(df, path_to_models, models, n_genes, threshold, logger_name, file_name, output_path, n_jobs):
    '''
    Select parameters for models, train models and calculate metrics

    :param df: dataframe with scRNA data
    :param path_to_models: str, path to the folder with models
    :param models: list, names of the models to train
    :param n_genes: int, number of the most important genes to take into account
    :param threshold: int, 1 - if we want to use adaptive threshold, else 0
    :param logger_name: str, name of created logger
    :param file_name: str, name of the file with scRNA-seq data (without extension)
    :param output_path: str, path to the folder with results
    :param n_jobs: int, the number of jobs to run in parallel
    '''

    logger = logging.getLogger(logger_name)
    # labels_dict = {cell_type: i for i, cell_type in enumerate(df.CellType.unique())}
    X = df.drop('CellType', axis=1)
    y = df['CellType']
    # y = df['CellType'].map(labels_dict)
    if not Path(path_to_models).exists():
        Path(path_to_models).mkdir(parents=True, exist_ok=True)

    metrics_dict = {'accuracy': {}, 'precision': {}, 'recall': {}, 'f1_score': {}}

    for model in models:
        X_filtered = dataset_processing(X=X, file_name=file_name, model=model,
                                        path_to_models=path_to_models,
                                        n_genes=n_genes, output_path=output_path,
                                        selected_genes=False)
        X_train_non_scaled, X_test_non_scaled, y_train, y_test = train_test_split(X_filtered, y, test_size=0.33)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_non_scaled)
        X_test_scaled = scaler.transform(X_test_non_scaled)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        if model == 'svc':
            X_train = X_train_scaled.copy()
            X_test = X_test_scaled.copy()
            grid_search_clf = RandomizedSearchCV(estimator=SVC(), param_distributions=PARAMS_SVC,
                                                 cv=skf, n_jobs=n_jobs, scoring='f1_macro')

        elif model == 'rforest':
            X_train = X_train_non_scaled.copy()
            X_test = X_test_non_scaled.copy()
            grid_search_clf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=PARAMS_RF,
                                                 cv=skf, n_jobs=n_jobs, scoring='f1_macro')

        elif model == 'lgmb':
            X_train = X_train_non_scaled.copy()
            X_test = X_test_non_scaled.copy()
            grid_search_clf = RandomizedSearchCV(estimator=LGBMClassifier(verbose=-1, n_jobs=n_jobs),
                                                 param_distributions=PARAMS_BOOSTING, cv=skf,
                                                 n_jobs=n_jobs, scoring='f1_macro')
        elif model == 'corr' and not threshold:
            X_train = X_train_scaled.copy()
            X_test = X_test_scaled.copy()
            grid_search_clf = CorrelationsClassifier()

        # model training
        logger.info(f'Start {model} training...')
        start_time = time.time()
        grid_search_clf.fit(X_train, y_train)
        if model != 'corr':
            best_clf = grid_search_clf.best_estimator_
        else:
            best_clf = grid_search_clf

        if threshold:
            best_clf = CalibratedThresholdsClassifier(best_clf, cv=3)
            best_clf.fit(X_train, y_train)

        y_test_pred = best_clf.predict(X_test)
        end_time = time.time()
        total_time = timedelta_to_hms(end_time-start_time)
        logger.info(f'End {model} raining.')
        if model != 'corr':
            logger.info(f'Best params: {grid_search_clf.best_params_}')
        logger.info(f'Total time for {model} training: {total_time}\n\n')

        # metrics calculation
        metrics_dict['accuracy'][model] = accuracy_score(y_test, y_test_pred)

        metrics_dict['precision'][model] = precision_score(y_test, y_test_pred, average='macro')

        metrics_dict['recall'][model] = recall_score(y_test, y_test_pred, average='macro')

        metrics_dict['f1_score'][model] = f1_score(y_test, y_test_pred, average='macro')

        pickle.dump(best_clf, open(f"{str(Path(path_to_models, model))}_{n_genes}.pkl", 'wb'))

    save_metrics(metrics_dict=metrics_dict, output_path=output_path, file_name=file_name)


def benchmarking(df, path_to_models, models, n_genes, logger_name, file_name, output_path):
    '''
    Train models and calculate metrics

    :param df: dataframe with scRNA data
    :param path_to_models: str, path to the folder with models
    :param models: list, names of the models to train
    :param n_genes: int, number of the most important genes to take into account
    :param logger_name: str, name of created logger
    :param logger_name: str, name of created logger
    :param file_name: str, name of the file with scRNA-seq data (without extension)
    :param output_path: str, path to the folder with results
    '''

    logger = logging.getLogger(logger_name)
    # labels_dict = {cell_type: i for i, cell_type in enumerate(df.CellType.unique())}
    X = df.drop('CellType', axis=1)
    y = df['CellType']
    # y = df['CellType'].map(labels_dict)
    if not Path(path_to_models).exists():
        Path(path_to_models).mkdir(parents=True, exist_ok=True)
    metrics_dict = {'accuracy': {}, 'precision': {}, 'recall': {}, 'f1_score': {}}
    logger.info('Start benchmarking...')
    for model in models:
        logger.info(f'Start model {model}...')
        X_filtered = dataset_processing(X=X, file_name=file_name, model=model,
                                        path_to_models=path_to_models,
                                        n_genes=n_genes, output_path=output_path,
                                        selected_genes=True)
        X_non_scaled = X_filtered.copy()

        path_to_model = Path(path_to_models, f'{model}_{n_genes}.pkl')
        if not path_to_model.exists():
            logger.info(f'{str(path_to_model)} does not exist!')
            sys.exit()
        best_clf = pickle.load(open(path_to_model, "rb"))

        if model in ['rforest', 'lgbm']:
            y_pred = best_clf.predict(X_non_scaled)

        elif model in ['svc', 'corr']:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_non_scaled)
            y_pred = best_clf.predict(X_scaled)

        # metrics calculation
        metrics_dict['accuracy'][model] = accuracy_score(y, y_pred)

        metrics_dict['precision'][model] = precision_score(y, y_pred, average='macro')

        metrics_dict['recall'][model] = recall_score(y, y_pred, average='macro')

        metrics_dict['f1_score'][model] = f1_score(y, y_pred, average='macro')
        logger.info(f'End model {model}...')

    save_metrics(metrics_dict=metrics_dict, output_path=output_path, file_name=file_name)
    logger.info('End benchmarking.')


def predict(df, path_to_models, models, n_genes, logger_name, file_name, output_path):
    '''
    Predict cell types for passed dataset

    :param df: dataframe with scRNA data
    :param path_to_models: str, path to the folder with models
    :param models: list, names of the models to train
    :param n_genes: int, number of the most important genes to take into account
    :param logger_name: str, name of created logger
    :param logger_name: str, name of created logger
    :param file_name: str, name of the file with scRNA-seq data (without extension)
    :param output_path: str, path to the folder with results
    '''

    logger = logging.getLogger(logger_name)
    X = df.copy()
    if not Path(path_to_models).exists():
        Path(path_to_models).mkdir(parents=True, exist_ok=True)
    for model in models:
        logger.info(f'Start prediction by {model}...')
        X_filtered = dataset_processing(X=X, file_name=file_name, model=model,
                                        path_to_models=path_to_models,
                                        n_genes=n_genes, output_path=output_path,
                                        selected_genes=True)
        X_non_scaled = X_filtered.copy()
        path_to_model = Path(path_to_models, f'{model}_{n_genes}.pkl')
        if not path_to_model.exists():
            logger.info(f'{str(path_to_model)} does not exist!')
            sys.exit()
        best_clf = pickle.load(open(path_to_model, "rb"))

        if model in ['rforest', 'lgbm']:
            y_pred = best_clf.predict(X_non_scaled)

        elif model in ['svc', 'corr']:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_non_scaled)
            y_pred = best_clf.predict(X_scaled)

        df_cell_types_pred = pd.DataFrame(y_pred, columns=['CellType'])
        df_cell_types_pred.to_csv(str(Path(output_path, file_name + f'_{model}_{n_genes}_cell_types.csv')), index=False)
        visualize_cell_types(dataset=df_cell_types_pred, model=model, n_genes=n_genes,
                             file_name=file_name, output_path=output_path)
        logger.info(f'End prediction by {model}')
