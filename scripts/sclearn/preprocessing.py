from typing import Optional

import numpy as np
from scipy.linalg import fractional_matrix_power

from sclearn.utils import add_noise


class DCA:
    """Discriminative Component Analysis data transformer

    Python implementaion of code from R package dml: https://rdrr.io/cran/dml/src/R/dca.R
    Reference publication: https://ink.library.smu.edu.sg/sis_research/2392/

    Parameters
    ----------
    dimensions : Optional[int] = None
        When not provided, DCA runs on the original data dimension.
        When given, dimension is reduced to `dimensions`
    noise_std : float = 1e-9
        If set, the random normal noise with standard deviation `noise_std` is added to the data
        during the fitting of transformation matrix. It helps to avoid multicollinearity
    debug : bool = False
        If set True, some debug messages are thrown as warnings

    """

    def __init__(self, dimensions: Optional[int] = None, noise_std: float = 1e-9, debug: bool = False):
        self.dimensions = dimensions
        self.noise_std = noise_std
        self.debug = debug

        self.dca = None
        self.transformed_data = None
        self.mahalanobis_matrix = None
        self.dca_transformation_matrix = None
        self.classes = None

    def fit(self, X, y, negative_links=None):
        """
        Run discriminative component analysis on the data

        Parameters
        ----------
        X : 2D array-like
            (n, d) shape matrix, where n is the number of data points and d is the dimension of the data
        y : Optional array-like
            Vector of length n containing classes for each data point. -1 means, that class is unknown
        negative_links : Optional 2D array-like
            (s, s) shape matrix, describing the negative relationships between classes. 1 in the position
            (i, j) means that i'th and j'th classes have negative constraints. 0 means, that classes
            do not have negative constraints or we have no information about that

        Returns
        -------
        result : namedtuple with fields B, DCA, data,
        - result.B: DCA suggested Mahalanobis matrix
        - result.DCA: DCA transormation of data
        - result.data: DCA transformed data

        For every two original data points (x1, x2) in result.data (y1, y2):
        (x2 - x1).T @ B @ (x2 - x1) = || (x2 - x1) @ A ||^2 = || y2 - y1 ||^2
        """
        if self.debug:
            from icecream import ic

        import warnings

        # Sometimes floats representation problems give complex numbers, when calculating eigenvalues
        # Here, we simply cast such complex numbers to floats. It causes ComplexWarning, which we supress
        warnings.filterwarnings("ignore", category=np.ComplexWarning)

        if self.noise_std:
            data = add_noise(X.T, std=self.noise_std)
        else:
            data = X.T

        d, n = data.shape

        if self.dimensions is None or self.dimensions < 1 or self.dimensions > d:
            self.dimensions = d

        self.classes = np.unique(y[y != -1])
        n_classes = len(self.classes)

        if negative_links is None:
            negative_links = np.ones(shape=(n_classes, n_classes))
            np.fill_diagonal(negative_links, 0)

        # 1. Compute means for each class
        classes_means = np.zeros(shape=(d, n_classes))

        for i, class_name in enumerate(self.classes):
            class_observations = (y == class_name)
            classes_means[:, i] = np.mean(data[:, class_observations], axis=1)

        if self.debug:
            ic(self.classes)
            ic(n_classes)
            ic(classes_means)

        # 2. Compute Cb
        Cb = np.zeros(shape=(d, d))

        N_d = 0

        for j, class_name in enumerate(self.classes):
            negatives = np.where(negative_links[j, :] == 1)[0]

            for i in range(len(negatives)):
                col = (classes_means[:, j] - classes_means[:, negatives[i]]).reshape(-1, 1)

                Cb = Cb + col @ col.T

            N_d += sum(negatives)

        if N_d:
            Cb = Cb / N_d
        else:
            Cb = np.eye(d)

        if self.debug:
            ic(Cb)
            ic(N_d)

        # 3. Compute Cw

        Cw = np.zeros(shape=(d, d))
        N_w = 0

        for j, class_name in enumerate(self.classes):
            class_observations = np.where(y == class_name)[0]

            for i in range(len(class_observations)):
                col = (data[:, class_observations[i]] - classes_means[:, j]).reshape(-1, 1)
                Cw = Cw + col @ col.T

            N_w += sum(class_observations)

        if N_w:
            Cw = Cw / N_w
        else:
            Cw = np.eye(d)

        if self.debug:
            ic(Cw)
            ic(N_w)

        # 3. Diagonalize Cb
        Cb_eigenvalues, Cb_eigenvectors = np.linalg.eig(Cb)
        Cb_eigenvalues = Cb_eigenvalues.reshape(-1, 1).astype(float)
        Cb_eigenvectors = Cb_eigenvectors.astype(float)  # Remove complex numbers caused by precision problems

        non_zero_values = (np.abs(Cb_eigenvalues) > 1e-9).flatten()

        R = Cb_eigenvectors[:, non_zero_values]
        R = R[:, : self.dimensions]

        Db = R.T @ Cb @ R
        Z = R @ fractional_matrix_power(Db, -0.5)

        if self.debug:
            ic(Cb_eigenvalues[:10])
            ic(Cb_eigenvectors)
            ic(R)
            ic(Db)
            ic(Z)

        # Diagonalize Z.T @ Cw @ Z
        Cz = Z.T @ Cw @ Z

        Cz_eigenvalues, Cz_eigenvectors = np.linalg.eig(Cz)
        non_zero_values = (np.abs(Cz_eigenvalues) > 1e-9).flatten()

        Cz_eigenvalues = Cz_eigenvalues[non_zero_values]
        Cz_eigenvectors = Cz_eigenvectors[:, non_zero_values]

        Dw_inverse_root = np.diag(np.power(Cz_eigenvalues, -0.5))

        if self.debug:
            ic(Cz)
            ic(Cz_eigenvalues[:10])
            ic(Cz_eigenvalues[-10:])
            ic(Cz_eigenvectors)
            ic(Dw_inverse_root)

        self.dca_transformation_matrix = Dw_inverse_root @ Cz_eigenvectors.T @ Z.T
        self.mahalanobis_matrix = self.dca_transformation_matrix.T @ self.dca_transformation_matrix

        if self.debug:
            ic(self.dca_transformation_matrix)
            ic(self.mahalanobis_matrix)

    def transform(self, X):
        return X @ self.dca_transformation_matrix.T
