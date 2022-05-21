from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from sclearn.preprocessing import DCA
from sclearn.utils import cluster_centroids, learn_thresholds


class CorrelationsClassifier:
    UNCERTAIN_CLASS_NAME = "unassigned"

    def _fit_DCA(self, X, y, negative_links):
        self.data_transformer = DCA(self.dimensions, self.noise_std, self.debug)
        self.data_transformer.fit(X, y, negative_links)

        transformed_data = self.data_transformer.transform(X)

        return transformed_data

    def _fit_identical_transformation(self, X):
        from sklearn.preprocessing import FunctionTransformer

        self.data_transformer = FunctionTransformer(lambda x: x)

        return X

    def __init__(self, correlation_method="pearson", data_transform_method="DCA", noise_std=1e-4, debug=False):
        self.correlation_method = correlation_method
        self.debug = debug
        self.data_transform_method = data_transform_method
        self.noise_std = noise_std

        self.dimensions = None
        self.data_transformer = None
        self.cell_types = None
        self.classes_centroids = None
        self.thresholds = []

    def fit(self, X, y, negative_links=None, dimensions: Optional[int] = None):
        self.dimensions = dimensions
        self.cell_types = np.unique(y)

        if self.data_transform_method == "DCA":
            transformed_data = self._fit_DCA(X, y, negative_links)
        elif self.data_transform_method is None:
            transformed_data = self._fit_identical_transformation(X)
        else:
            raise ValueError(f"{self.data_transform_method} is not a valid transform method")

        self.classes_centroids = cluster_centroids(y, transformed_data)
        self.thresholds = np.array(list(
            learn_thresholds(
                y, transformed_data, self.classes_centroids, correlation_method=self.correlation_method).values()
        ))

    def predict(self, X):
        transformed_data = self.data_transformer.transform(X)

        predicted_cell_types = []

        for cell in transformed_data:
            max_corr = -np.inf
            best_type_idx = None

            for i, cell_type in enumerate(self.cell_types):
                correlation = np.corrcoef(cell, self.classes_centroids[cell_type])[0, 1]

                if correlation > max_corr and correlation > self.thresholds[i]:
                    max_corr = correlation
                    best_type_idx = i

            if best_type_idx is None:
                best_type = self.UNCERTAIN_CLASS_NAME
            else:
                best_type = self.cell_types[best_type_idx]

            predicted_cell_types.append(best_type)

        return np.array(predicted_cell_types).astype(str)


class CalibratedThresholdsClassifier:
    UNCERTAIN_CLASS_NAME = -1

    def __init__(self, base_estimator, cv=None, calibration_method="sigmoid", percentile=1):
        self.base_estimator = base_estimator
        self.calibration_method = calibration_method
        self.cv = cv
        self.percentile = percentile

        self.calibrated_classifier = CalibratedClassifierCV(base_estimator=self.base_estimator, cv=self.cv,
                                                            method=self.calibration_method)

        self.thresholds = []
        self.cell_types = None

    def fit(self, X, y):
        self.cell_types = np.unique(y)
        self.calibrated_classifier.fit(X, y)

        probabilities = self.calibrated_classifier.predict_proba(X)

        for cell_type in self.cell_types:
            class_idx = np.where(self.calibrated_classifier.classes_ == cell_type)[0].item()
            class_probs = probabilities[y == cell_type, class_idx]

            self.thresholds.append(np.percentile(class_probs, self.percentile))

        self.thresholds = np.array(self.thresholds)

    def predict_proba(self, X):
        return self.calibrated_classifier.predict_proba(X)

    def predict(self, X):
        predicted_cell_types = np.array([self.UNCERTAIN_CLASS_NAME] * X.shape[0]).astype(self.cell_types.dtype)

        probabilities = self.predict_proba(X)
        best_classes = probabilities.argmax(axis=1)

        # I'm pretty sure it can be done better by some numpy function, but I didn't figure it out yet
        best_class_probs = np.array([prob[class_idx] for prob, class_idx in zip(probabilities, best_classes)])

        prob_exceeding_threshold = best_class_probs > self.thresholds[best_classes]
        predicted_cell_types[prob_exceeding_threshold] = self.cell_types[best_classes][prob_exceeding_threshold]

        return predicted_cell_types.astype(str)
