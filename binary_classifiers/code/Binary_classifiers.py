import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (make_scorer, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.linear_model import LinearRegression

import xgboost

from xgboost_autotune import fit_parameters

import warnings

warnings.simplefilter('ignore')


def rank_genes_train_set(X_train: pd.DataFrame, max_genes: int = None) -> tuple:
    """
    Rank genes based on residuals of the model mean_exp ~ dropout_rate
    """
    dropout = (X_train == 0).sum(axis=0)
    dropout = (dropout / X_train.shape[0]) * 100
    mean = X_train.mean(axis=0)

    notzero = np.where((np.array(mean) > 0) & (np.array(dropout) > 0))[0]
    zero = np.where(~((np.array(mean) > 0) & (np.array(dropout) > 0)))[0]
    train_notzero = X_train.iloc[:, notzero]
    train_zero = X_train.iloc[:, zero]
    zero_genes = train_zero.columns

    dropout = dropout.iloc[notzero]
    mean = mean.iloc[notzero]

    #  dropout = np.log2(np.array(dropout)).reshape(-1, 1)
    dropout = np.array(dropout).reshape(-1, 1)
    mean = np.array(mean).reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(mean, dropout)

    residuals = dropout - reg.predict(mean)
    residuals = pd.Series(np.array(residuals).ravel(), index=train_notzero.columns)
    residuals = residuals.sort_values(ascending=False)
    sorted_genes = residuals.index
    sorted_genes = list(sorted_genes.append(zero_genes))

    return residuals, sorted_genes[:max_genes]


class BinaryClassifiersMaker():

    def __init__(self,
                 verbose=False):

        """
        Initializing classifier maker. 
        
        Args:
            verbose: if set True, all information messages will be printed
            
        Returns:
            An example of classifier maker.
        """

        self.verbose = verbose
        self.classifiers = {}  # All created classifiers will be stored in this dict as
        # {celltype : [[clf, clf_genes:list] * number_of_classifiers] * number_of_celltypes}

        self.current_cell_type_and_dataset = None  # Current classifier metadata
        self.current_X_test = None  # Current classifier metadata
        self.current_y_test = None  # Current classifier metadata
        self.current_clf = None  # Current classifier

    def read_new_dataset(self,
                         dataset,
                         celltype_column_name="CellType"):

        """
        Reads new dataset from pandas Dataframe into object workspace.
        
        Args:
            dataset: name of pandas Dataframe
            celltype_column_name: name of column with names of celltypes
        """

        self.dataset = dataset  # Saves dataset variable to work with it
        self.selected_genes_number = None  # For dataset in work via fucntion rank_genes_manually
        self.cell_types = dataset[celltype_column_name].unique()  # Saves all celltypes of this dataset

        self.X = dataset.drop(celltype_column_name, axis=1)  # Feature matrix
        self.y = dataset[celltype_column_name]  # Target

        self.residuals, self.sorted_genes = rank_genes_train_set(X_train=self.X)  # For function rank_genes_manually

        return

    def read_new_dataset_from_csv(self,
                                  filename,
                                  index_column="cells",
                                  celltype_column_name="CellType"):

        """
        Reads new dataset from .csv file into object workspace. 
        Based on read_new_dataset function
        
        Args:
            filename: path to file
            index_column: columns with cell names
            celltype_column_name: name of column with names of celltypes
        """

        if self.verbose is True: print("loading dataset")
        dataset = pd.read_csv(filename, index_col=index_column)
        if self.verbose is True: print("dataset loaded")
        self.read_new_dataset(dataset, celltype_column_name=celltype_column_name)
        return

    def _show_ranked_genes(self, residuals):

        """
        Adaptively shows plot of ranked genes. Used in rank_genes_manually function, 
        sets self.selected_genes_number value.
        """

        maximum = 2000
        while maximum != "ok":
            plt.figure(figsize=(12, 8))
            plt.plot(np.arange(maximum), residuals[:maximum], linewidth=5)
            plt.xlabel('ranked genes', fontsize=20)
            plt.ylabel('residuals', fontsize=20)
            plt.axvline(maximum // 2, linestyle='--', color='red')
            plt.text(maximum // 2, 10, f'{maximum // 2}', rotation=90, fontsize=14)
            plt.axvline(maximum // 3, linestyle='--', color='red')
            plt.text(maximum // 3, 10, f'{maximum // 3}', rotation=90, fontsize=14)
            plt.axvline(maximum // 4, linestyle='--', color='red')
            plt.text(maximum // 4, 10, f'{maximum // 4}', rotation=90, fontsize=14)
            plt.show()
            new_maximum = input("Please enter another maximum to zoom in/out. If you are satisfied, enter 'ok' \n")
            if new_maximum.isdigit():
                maximum = int(new_maximum)
                plt.close()
                continue
            elif new_maximum != "ok":
                print("Your entered inappropriate input. Saved previous maximum value")
            else:
                maximum = new_maximum

        while True:
            selected_genes_number = input("Enter desired genes number:    ")
            if selected_genes_number.isdigit():
                return int(selected_genes_number)
            print("You have entered inappropriate value. Please enter int value")

    def rank_genes_manually(self):

        """
        Visualises interactive plot in ranked genes - residuals coordinates.
        Genes with residuals before plato are recommended to choose. Sets self.selected_genes_number
        as the result of function call. Based on show_ranked_genes function (out of the class)
        """

        if self.verbose is True:
            print("Starting manual genes ranking")
        selected_genes_number = self._show_ranked_genes(self.residuals)
        self.selected_genes_number = selected_genes_number
        print(f"selected_genes_number default is set to {selected_genes_number}")
        return

    def make_classifier(self,
                        chosen_celltype,
                        selected_genes_number=None,
                        scale=False,
                        balance_classes=True,
                        metrics_threshold=0.7,
                        dataset_number=None):

        """
        Makes binary classifier (based on Xgboost) for one celltype for this (current self.dataframe) 
        in work. Loads classifier and classifiers' features (genes) in self.classifiers[celltype].
        Sets new self.current_clf, self.current_X_test, self.current_y_test.
        
        
        Args:
            chosen_celltype - selected celltype for current dataset
            
            selected_genes_number - manually set genes number in ranked genes (if not specified,
            self.selected_genes_number will be used.
            If both selected_genes_number and self.selected_genes_number are not specified, raises ValueError
            
            scale - scale dataset with standard scaler
        
            balance_classes - upsamples the low-established class if it accounts for less than 10% of cells
            
            metrics_threshold - if the lowest metric (accuracy, precision, recall, f1_score) on
             a test (for this celltype in current dataset)
            is below threshold, this classifier will not be added to self.classifiers
        
            dataset_number - for verbose needs
            
        Returns:
            classifier
        """

        self.current_cell_type_and_dataset = chosen_celltype, self.dataset

        labels = {cell_type: (1 if cell_type == chosen_celltype else 0) for cell_type in self.cell_types}
        y = self.y.map(labels)

        if selected_genes_number is not None:
            selected_genes = self.sorted_genes[:selected_genes_number]
        elif self.selected_genes_number is not None:
            selected_genes = self.sorted_genes[:self.selected_genes_number]
        else:
            raise ValueError(
                'You have not specified seleceted_genes_number. Please use rank_genes_manually() '
                'or specify selected_genes_number')
        X_filtered = self.X[selected_genes]

        if balance_classes is True:

            classes_count = np.unique(y, return_counts=True)
            if self.verbose is True:
                print(f"Balance of classes before upsampling is {classes_count}")

            percentage_lower_class = np.min(classes_count[1]) / np.max(classes_count[1])

            if percentage_lower_class < 0.1:

                df = X_filtered
                df["y"] = y
                while percentage_lower_class < 0.1:
                    df_max = df.loc[df["y"] == np.argmax(classes_count[1])]
                    df_min = df.loc[df["y"] == np.argmin(classes_count[1])]
                    df_one = df_min.copy()
                    df_two = df_min.copy()
                    new_df_min = pd.concat([df_one, df_two])
                    df = pd.concat([df_max, new_df_min])
                    df = df.sample(frac=1)
                    classes_count = np.unique(df["y"], return_counts=True)
                    percentage_lower_class = np.min(classes_count[1]) / np.max(classes_count[1])

                X_filtered = df.drop("y", axis=1)
                y = df["y"]
                if self.verbose is True:
                    print(f"Balance of classes after upsampling is {np.unique(y, return_counts=True)}")

            else:
                if self.verbose is True:
                    print("Balance of classes is good, no need for upsampling")

        X_train, X_test, y_train, y_test = train_test_split(X_filtered,
                                                            y, stratify=y,
                                                            test_size=0.3,
                                                            random_state=42)
        if scale is True:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        self.current_X_test = X_test
        self.current_y_test = y_test

        if self.verbose is True:
            print("Performing Xgboost \n")
        accuracy = make_scorer(accuracy_score, greater_is_better=True)
        fitted_model = fit_parameters(initial_model=xgboost.XGBRegressor(),
                                      initial_params_dict={},
                                      X_train=X_train,
                                      y_train=y_train,
                                      min_loss=0.01,
                                      scoring=accuracy,
                                      n_folds=5)

        params = fitted_model.get_params()
        classifier = xgboost.XGBClassifier()
        classifier.set_params(**params)
        classifier.fit(X_train, y_train)
        self.current_clf = classifier

        y_pred = classifier.predict(X_test)
        scores = np.array([accuracy_score(y_test, y_pred),
                           precision_score(y_test, y_pred),
                           recall_score(y_test, y_pred),
                           f1_score(y_test, y_pred)])

        if self.verbose is True:
            print(f"Metrics for {chosen_celltype} of dataset {dataset_number}."
                  f"accuracy: {scores[0]}, precision: {scores[1]}, recall: {scores[2]}, f1-score: {scores[3]}")
        if np.sum(scores > metrics_threshold) == len(scores):

            if f"{chosen_celltype}" not in self.classifiers:
                self.classifiers[f"{chosen_celltype}"] = [[classifier, selected_genes]]
            else:
                self.classifiers[f"{chosen_celltype}"].append([classifier, selected_genes])

        else:
            if self.verbose is True:
                if dataset_number is not None:
                    print(
                        f"classifier for {chosen_celltype} of dataset number {dataset_number} "
                        f"did not passed metrics filter")
                else:
                    print(f"classifier for {chosen_celltype} did not passed metrics filter")

        return classifier

    def show_metrics(self):

        """Shows metrics on a test (for this celltype in current dataset)"""

        y_pred = self.current_clf.predict(self.current_X_test)
        print(f"Showing metrics for binary classificator based on {self.current_cell_type_and_dataset[0]} celltype: \n")
        print("accuracy:", accuracy_score(self.current_y_test, y_pred), "\n",
              "precison:", precision_score(self.current_y_test, y_pred), "\n",
              "recall:", recall_score(self.current_y_test, y_pred), "\n",
              "f1-score:", f1_score(self.current_y_test, y_pred), sep="")
        return

    def show_currents(self):

        """
        Shows current dataset shape and all datasets' celltypes. Show current celltype,
        which is the base of current classifier
        """

        print(f"Current dataset has shape {self.dataset.shape} with celltypes:")
        print(*self.cell_types, sep=", ")
        if self.current_cell_type_and_dataset is not None:
            print(f"Current cell type: {self.current_cell_type_and_dataset[0]}")
        return

    def return_clf_dict(self):

        """returns self.classifiers"""

        return self.classifiers

    def process_whole_dataset(self,
                              selected_genes_number=None,
                              min_class_threshold=5,
                              metrics_threshold=0.7,
                              balance_classes=True,
                              scale=False,
                              dataset_number=None):

        """
        Makes binary classifiers (based on Xgboost) for all celltypes of this dataset
        
        Args:
            selected_genes_number - manually set genes number in ranked genes (if not specified,
            self.selected_genes_number will be used.
            If both selected_genes_number and self.selected_genes_number are not specified, raises ValueError.
            Argument passes into make_classifier function
            
            min_class_threshold - minimum number of cells in low established class for celltype. If the number is
            below threshold, classifier based on this celltype will not be added to self.classifiers
            
            metrics_threshold - if the lowest metric (accuracy, precision, recall, f1_score) on a test
            (for this celltype in current dataset)
            is below threshold, this classifier will not be added to self.classifiers.
            Argument passes into make_classifier function
            
            balance_classes - upsamples the low-established class if it accounts for less than 10% of cells. 
            Argument passes into make_classifier function
        
            scale - scale dataset with standard scaler. Argument passes into make_classifier function
            
            dataset_number - for verbose needs. Argument passes into make_classifier function
        """

        for chosen_cell_type in self.cell_types:

            if dataset_number is not None:
                if self.verbose is True:
                    print(f"Working with {chosen_cell_type} of dataset {dataset_number}")
            else:
                if self.verbose is True:
                    print(f"Working with {chosen_cell_type}")

            labels = {cell_type: (1 if cell_type == chosen_cell_type else 0) for cell_type in self.cell_types}
            y = self.y.map(labels)

            threshold_data = np.unique(y, return_counts=True)[1]
            threshold = np.min(threshold_data)

            if threshold < min_class_threshold:
                if self.verbose is True:
                    if dataset_number is not None:
                        print(
                            f"class balance for celltype {chosen_cell_type} of dataset number {dataset_number} "
                            f"is below threshold. Skipping this celltype")
                    else:
                        print(
                            f"class balance for celltype {chosen_cell_type} is below threshold. Skipping this celltype")
                continue

            self.make_classifier(chosen_cell_type,
                                 selected_genes_number=selected_genes_number,
                                 scale=scale,
                                 metrics_threshold=metrics_threshold,
                                 balance_classes=balance_classes,
                                 dataset_number=dataset_number)
        return

    def fit(self, *args,
            mode="files",
            min_number_cells=100,
            min_class_threshold=5,
            metrics_threshold=0.7,
            balance_classes=True,
            manual_genes_rank=True,
            selected_genes_number=None,
            scale=False):

        """
        Makes binary classifiers (based on Xgboost) for all datasets given
        
        Args:
            mode: dataframes if pandas dataframes are given, files if pathways to .CSVs are given
            
            min_number_cells: check the number of cells in dataset. If number of cells is below 
            minimum, this dataset will be skipped
            
            min_class_threshold: minimum number of cells in low established class for celltype. If the number is
            below threshold, classifier based on this celltype will not be added to self.classifiers.
            Argument passes into process_whole_dataset function
            
            metrics_threshold: if the lowest metric (accuracy, precision, recall, f1_score)
            on a test (for this celltype in current dataset) is below threshold, this classifier
            will not be added to self.classifiers. 
            Argument passes into process_whole_dataset fucntion => make_classifier function
            
            balance_classes: upsamples the low-established class if it accounts for less than 10% of cells.
            Argument passes into process_whole_dataset fucntion => make_classifier function
            
            manual_genes_rank: will call rank_genes_manually for each dataset given.
            
            selected_genes_number: manually set genes number in ranked genes for all datasets. Will
            automatically set manual_genes_rank as False. 
            Argument passes into process_whole_dataset fucntion => make_classifier function

            scale: scale dataset with standard scaler. Argument passes into process_whole_dataset fucntion =>
            make_classifier function
        """

        if selected_genes_number is not None:
            manual_genes_rank = False

        if mode == "files":
            count = 1
            for arg in args:
                if self.verbose is True:
                    print(f"Reading  dataset {count}")
                self.read_new_dataset_from_csv(arg)
                if self.dataset.shape[0] < min_number_cells:
                    if self.verbose is True:
                        print(f" dataset {count} has less than {min_number_cells} cells. Skipping this dataset")
                    count += 1
                    continue
                if manual_genes_rank is True:
                    self.rank_genes_manually()
                self.process_whole_dataset(selected_genes_number=selected_genes_number,
                                           min_class_threshold=min_class_threshold,
                                           metrics_threshold=metrics_threshold,
                                           balance_classes=balance_classes,
                                           scale=scale,
                                           dataset_number=count)
                count += 1
        elif mode == "dataframes":
            count = 1
            for arg in args:
                if self.verbose is True:
                    print(f"Working with  dataset {count}")
                self.read_new_dataset(arg)
                if self.dataset.shape[0] < min_number_cells:
                    if self.verbose is True:
                        print(f" dataset {count} has less than {min_number_cells} cells. Skipping this dataset")
                    count += 1
                    continue
                if manual_genes_rank is True:
                    self.rank_genes_manually()
                self.process_whole_dataset(selected_genes_number=selected_genes_number,
                                           min_class_threshold=min_class_threshold,
                                           metrics_threshold=metrics_threshold,
                                           balance_classes=balance_classes,
                                           scale=scale,
                                           dataset_number=count)
                count += 1
        return

    def _predict_proba_one_clf(self,
                               dataset,
                               trained_clf,
                               columns):

        """
        predicts probes for one classifier for dataset given. 
        
        Args:
            dataset: test dataset for prediction
            trained_clf: classifier for prediction
            columns: features of this classifier
            
        Returns:
            predicted proba values for celltype of trained classifier
        """

        dataset_transformed = dataset[columns]
        y_pred = trained_clf.predict_proba(dataset_transformed).T[1]  # предсказывает вероятность единицы -
        # что это именно этот клеточный тпип
        return y_pred

    def predict_proba_for_cell_type(self,
                                    dataset,
                                    celltype,
                                    method="average"):

        """
        Predict probes for this celltype for each classifier. Averages probes of all
        classifiers
        
        Args:
            dataset: test dataset for prediction
            
            celltype: celltype of classifiers
            
            method: average - averaging probes of all classifiers, raw - do not average
            
        Returns:
            averaged probes (average)
            dataframe of probes for each classifier (raw)
        """

        classifiers_columns = self.classifiers[celltype]
        probes = {}
        count = 1
        for classifier, columns in classifiers_columns:
            probes[f"{celltype}_clf_{count}"] = self._predict_proba_one_clf(dataset, classifier, columns)
            count += 1
        probes = pd.DataFrame(probes)
        if method == "average":
            probes[f"{celltype}"] = probes.mean(axis=1)
            return probes[f"{celltype}"]
        elif method == "raw":
            return probes

    def predict_proba_for_all_trained_celltypes(self,
                                                dataset,
                                                adaptive_threshold=False):

        """
        Predict average probes for all celltypes presented in self.classifiers
        
        Args:
            dataset: test dataset for prediction
            
            adaptive_threshold: Choose threshold of classifying for each celltypes. If proba 
            for this celltype is below threshold, this proba will be changed to 0 
            
        Returns:
            dataframe with probes for celltypes
        """

        probes = {}
        cell_types = self.classifiers.keys()
        for cell_type in cell_types:
            probes[f"{cell_type}"] = self.predict_proba_for_cell_type(dataset, cell_type, method="average")
        probes = pd.DataFrame(probes)
        if adaptive_threshold is True:
            for cell_type in probes.columns:
                threshold = np.percentile(probes[cell_type], 1)
                probes[probes[cell_type] < threshold] = 0
        return probes

    def predict(self,
                dataset,
                mode="dataframe",
                index_column="cells",
                adaptive_threshold=False):

        """
        Predicts celltypes for given dataset. Computes probes for each celltype presented in 
        self.classifiers (predict_proba_for_all_trained_celltypes function). Chooses the highest
        proba for current cell. 
        
        Args:
            dataset: test dataset for prediction
            
            mode: if dataframe, provide pandas Dataframe, elif file, provide path to .csv file
            
            index_column: index column of .csv
            
            adaptive_threshold: Choose threshold of classifying for each celltypes. Argument passes
            into predict_proba_for_all_trained_celltypes fucntion
            
        Returns:
            predictions for celltypes of test dataset
        """

        if mode == "file":
            dataset = pd.read_csv(dataset, index_col=index_column)
        probes = self.predict_proba_for_all_trained_celltypes(dataset, adaptive_threshold=adaptive_threshold)
        indexes = probes.apply(np.argmax, axis=1)
        values = probes.apply(np.max, axis=1)
        if adaptive_threshold is True:
            for idx in range(len(indexes)):
                if values[idx] == 0:
                    indexes[idx] = "indefinite"
        celltypes = probes.columns
        indexes_to_celltypes = {celltype: index for celltype, index in enumerate(celltypes)}
        indexes_to_celltypes["indefinite"] = "Unassigned"
        prediction = indexes.map(indexes_to_celltypes)
        return prediction

    def benchmark(self,
                  dataset,
                  mode="dataframe",
                  index_column="cells",
                  celltype_column_name="CellType",
                  adaptive_threshold=False):

        """
        Predicts celltypes for a given dataset and computemetrics compating predictions with real
        celltypes (target). Metrics are calculated in "macro" mode
        
        Args:
            dataset: test dataset for prediction
            
            mode: if dataframe, provide pandas Dataframe, elif file, provide path to .csv file
            
            index_column: index column of .csv
            
            celltype_column_name: name of column with celltypes(target)
            
            adaptive_threshold: Choose threshold of classifying for each celltypes. Argument passes
            into predict_proba_for_all_trained_celltypes fucntion
        
        Returns:
            metrics for dataset  
        """

        if mode == "file":
            dataset = pd.read_csv(dataset, index_col=index_column)
        X = dataset.drop(celltype_column_name, axis=1)
        y = dataset[celltype_column_name]
        prediction = self.predict(X, mode="dataframe",
                                  adaptive_threshold=adaptive_threshold)
        accuracy = accuracy_score(y, prediction)
        precision = precision_score(y, prediction, average="macro")
        recall = recall_score(y, prediction, average="macro")
        f1 = f1_score(y, prediction, average="macro")
        print(f"accuracy: {accuracy}",
              f"precision: {precision}",
              f"recall: {recall}",
              f"f1_score: {f1}", sep="\n")
        return

    def save_model(self, model_name):

        """
        Saves self.classifiers structure to .zip file in models folder
        
        Args:
            model_name: name of the saved .zip file
        """

        os.mkdir(f"../models/{model_name}")

        for celltype in self.classifiers.keys():
            os.mkdir(f"../models/{model_name}/{celltype}")
            for number, model in enumerate(self.classifiers[celltype]):
                pickle.dump(model, open(f"../models/{model_name}/{celltype}/clf_{number}", "wb"))
        shutil.make_archive(f"../models/{model_name}", 'zip', f"../models/{model_name}")
        shutil.rmtree(f"../models/{model_name}")
        return

    def load_model(self, model_name):

        """
        Loads self.classifiers from .zip file located in models folder
        
        Args:
            model_name: name of .zip file located in models folder
        """

        if len(self.classifiers) != 0:
            print("This maker already have model, cannot load another one")
            return
        shutil.unpack_archive(f"../models/{model_name}.zip", f"../models/{model_name}")
        for celltype in os.listdir(f"../models/{model_name}"):
            self.classifiers[celltype] = []
            for clf in os.listdir(f"../models/{model_name}/{celltype}"):
                self.classifiers[celltype].append(pickle.load(open(f"../models/{model_name}/{celltype}/{clf}", "rb")))
        shutil.rmtree(f"../models/{model_name}")
        return

    def add_classifiers_from_model(self, model_name):

        """
        Adds classifiers to existing celltypes in self.classifiers from model
        
        Args:
            model_name: name of .zip file located in models folder
        """

        if len(self.classifiers) == 0:
            print("Cannot add classifiers, current model is empty")
            return
        shutil.unpack_archive(f"../models/{model_name}.zip", f"../models/{model_name}")
        for celltype in os.listdir(f"../models/{model_name}"):
            if celltype in self.classifiers:
                for clf in os.listdir(f"../models/{model_name}/{celltype}"):
                    self.classifiers[celltype].append(
                        pickle.load(open(f"../models/{model_name}/{celltype}/{clf}", "rb")))
        shutil.rmtree(f"../models/{model_name}")
        return

    def add_celltypes_from_model(self, model_name):

        """
        Adds new celltypes to existing self.classifiers from model
        
        Args:
            model_name: name of .zip file located in models folder
        """

        shutil.unpack_archive(f"../models/{model_name}.zip", f"../models/{model_name}")
        for celltype in os.listdir(f"../models/{model_name}"):
            if celltype not in self.classifiers:
                self.classifiers[celltype] = []
                for clf in os.listdir(f"../models/{model_name}/{celltype}"):
                    self.classifiers[celltype].append(
                        pickle.load(open(f"../models/{model_name}/{celltype}/{clf}", "rb")))
        shutil.rmtree(f"../models/{model_name}")
        return

    def add_all(self, model_name):

        """
        Adds celltypes (with theie classifiers) and classifiers to existing celltypes in
        self.classifiers from model
        
        Args:
            model_name: name of .zip file located in models folder
        """

        shutil.unpack_archive(f"../models/{model_name}.zip", f"../models/{model_name}")
        for celltype in os.listdir(f"../models/{model_name}"):
            if celltype not in self.classifiers:
                self.classifiers[celltype] = []
            for clf in os.listdir(f"../models/{model_name}/{celltype}"):
                self.classifiers[celltype].append(pickle.load(open(f"../models/{model_name}/{celltype}/{clf}", "rb")))
        shutil.rmtree(f"../models/{model_name}")
        return

    def check_prediction_availibility(self, test_dataset):

        """
        For each classifier checks if this classifier can be applied to test dataset
        
        Args:
            test_dataset: dataset for check 
        """
        availibility = True
        dataset_genes = set(test_dataset.columns)
        for celltype in self.classifiers:
            count = 0
            for _, genes in self.classifiers[celltype]:
                genes = set(genes)
                if not genes.issubset(dataset_genes):
                    availibility = False
                    print(f"classifier with index {count} of {celltype} does not match test dataset genes")
                count += 1
        return availibility
