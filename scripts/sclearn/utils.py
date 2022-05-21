import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import seaborn as sns


def cluster_centroids(cell_types, cells_data) -> dict:
    """Find centroids for each cell type"""

    unique_types = np.unique(cell_types)

    centroids = {}  # Map from cluster names to the centroids

    for cell_type in unique_types:
        cluster_data = cells_data[cell_types == cell_type, :]
        centroids[cell_type] = cluster_data.mean(axis=0)

    return centroids


def clusters_cells_correlations(cells_data, centroids, correlation_method="pearson"):
    """Calculate correlation of each cell with clusters' centroids"""

    cells_correlations = []

    for cell in cells_data:
        if correlation_method == "pearson":
            cell_cluster_correlation = [pearsonr(cell, centroid)[0] for centroid in centroids.values()]
        elif correlation_method == "spearman":
            cell_cluster_correlation = [spearmanr(cell, centroid)[0] for centroid in centroids.values()]
        else:
            raise ValueError(f"Correlation method {correlation_method} is not supported")

        cell_cluster_correlation = np.array(cell_cluster_correlation)

        cells_correlations.append(cell_cluster_correlation)

    return np.array(cells_correlations)


def learn_thresholds(cell_types, cells_data, centroids, correlation_method="pearson"):
    thresholds = {}  # Map from cell types to the threshold

    correlations = clusters_cells_correlations(cells_data, centroids, correlation_method=correlation_method)

    for i, cell_type in enumerate(centroids.keys()):
        cluster_cells_correlations = correlations[cell_types == cell_type, i]
        thresholds[cell_type] = np.percentile(cluster_cells_correlations, 1)

    return thresholds


def feature_cluster(expression_profile: pd.DataFrame, sample_information: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean expression of each gene for each cell type

    Returns
    -------
    Table where rows are cell types and columns are genes with mean gene expression in each cluster
    """

    assert all(expression_profile.index == sample_information.index)

    expression_profile["cell_type"] = sample_information["cell_type"]

    return expression_profile.groupby("cell_type").aggregate("mean")


def add_noise(matrix, std=1e-4):
    """Add random normal noise to `matrix` to avoid multicollinearity"""
    return matrix + np.random.normal(0, std, size=matrix.shape)


def visualise_correlations(cells_data, cells_types, noise_std=1e-9, correlation_method="pearson"):
    from sclearn.preprocessing import DCA

    dca = DCA(noise_std=noise_std)
    dca.fit(cells_data, cells_types)

    transformed_data = dca.transform(cells_data)
    centroids = cluster_centroids(cells_types, transformed_data)

    cells_correlations = clusters_cells_correlations(transformed_data, centroids, correlation_method)

    for i, cell_type in enumerate(centroids.keys()):
        fig, axes = plt.subplots(figsize=(10, 5), ncols=2)

        sns.violinplot(x=cells_correlations[:, i], ax=axes[0], palette="muted")

        cluster_cells_correlations = cells_correlations[cells_types == cell_type, i]
        sns.violinplot(x=cluster_cells_correlations, ax=axes[1], palette="muted")

        axes[0].set_xlim(-1, 1)
        axes[1].set_xlim(-1, 1)

        axes[0].set_title(f"All cells similarity with {cell_type} centroid")
        axes[1].set_title(f"{cell_type} similarity with {cell_type} centroid")

        threshold = np.percentile(cluster_cells_correlations, 1)

        axes[0].axvline(threshold, color="pink")
        axes[1].axvline(threshold, color="pink")

        fig.suptitle(f"{cell_type}, threshold: {round(threshold, 4)}")
        fig.tight_layout()
        plt.show()
