from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_cell_types(dataset, model, n_genes, file_name, output_path):
    '''
    Visualize ditribution of cell types in dataset

    :param dataset: dataframe with scRNA-seq data
    :param model: str, name of the models to train
    :param n_genes: int, number of the most important genes to take into account
    :param file_name: str, name of the file with scRNA-seq data (without extension)
    :param output_path: str, path to folder with the results of visualization
    '''

    plt.figure(figsize=(12, 8))
    sns.countplot(x=dataset['CellType'])
    plt.title('Cell populations', fontsize=16)
    plt.xlabel('CellType', fontsize=14)
    plt.ylabel('count', fontsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.savefig(str(Path(output_path, f'{file_name}_{model}_{n_genes}_cell_types.png')), bbox_inches="tight")


def visualize_residuals(residuals, n_genes, output_path, file_name):
    '''
    Visualize residuals of the model mean_exp ~ dropout_rate

    :param residuals: list with residuals from the model mean_exp ~ dropout_rate
    :param n_genes: int, number of the most important genes to take into account
    :param output_path: str, path to folder with the results of visualization
    :param file_name: str, name of the file with scRNA-seq data (without extension)
    '''

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(residuals)), residuals)
    plt.axvline(n_genes, linestyle='--', color='red')
    plt.xlabel('ranked genes', fontsize=14)
    plt.ylabel('residuals', fontsize=14)
    plt.savefig(str(Path(output_path, file_name + '_ranked_genes.png')), bbox_inches="tight")
