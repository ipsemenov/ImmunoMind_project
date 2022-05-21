# Developing best practices for semi-automatic single-cell data annotation

**Single-cell sequencing** is paving the way for precision medicine. It is the next step towards making precision medicine more accurate. One of the most important step in single-cell data analysis is **cell type labeling**. This is a very time-consuming process, the automation of which is a task of current interest.

The goal of this project is using modern machine learning approaches to build semi-automatic single-cell data annotation tool.
        
Authors:
- Ivan Semenov
- Anton Muromtsev
- Vladimir Shitov 

Supervisors:
- Daniil Litvinov
- Vasily Tsvetkov

The project was made for the [ImmunoMind Inc](https://immunomind.io) as a part of the education at the [Bioinformatics Institute](https://bioinf.me/en/education).

# Graphical abstract

![Poster](images/poster.png)

# Aims
1. Mine public databases, to collect high-quality single-cell datasets
1. Preprocess datasets before building ML models
1. Build a pipeline for cell type annotation
1. Add possibility to label uncertain predictions as “Unassigned”
1. Create a benchmark of single-cell data annotation instruments

# Datasets
We used datasets from the work of Abdelaal et al [1]. Datasets are available via the [link](https://zenodo.org/record/3357167#.YokrjC8Rp-V). We evaluated our models on the following datasets:
- 10Xv2_pbmc1
- 10Xv3_pbmc1
- DR_pbmc1.csv

# Trained models

## Support Vector Machine Classifier

## Random Forest Classifier

## LightGBM Classifier

## Correlations Classifier

## Hierarchical binary classificators

# Labelling the unassigned cells

# References
1. Abdelaal, T., Michielsen, L., Cats, D. et al. A comparison of automatic cell identification methods for single-cell RNA sequencing data. Genome Biol 20, 194 (2019). https://doi.org/10.1186/s13059-019-1795-z
