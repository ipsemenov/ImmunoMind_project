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

# Results
- Collected datasets for benchmark
- Implemented  an approach to mark uncertain predictions with individual thresholds  for each cell type
- Trained and compared ML models for cell type classification:
- - Support Vector Machine Classifier
- - Random Forest Classifier
- - LightGBM Classifier
- - Hierarchical binary classificators
- - Classificator based on the correlations in the Discriminative PCA space

# Datasets
We used datasets from the work of Abdelaal et al [1]. Datasets are available via the [link](https://zenodo.org/record/3357167#.YokrjC8Rp-V). We evaluated our models on the following datasets:
- 10Xv2_pbmc1
- 10Xv3_pbmc1
- DR_pbmc1.csv

# Labelling the unassigned cells
Some classifiers can produce some certanty measure of the prediction (e.g. probabillity). It can be used to mark uncertain predictions. But how to define an optimal threshold for the classificator? Duan et al [2] suggested using the 1st percentile of the certanty for each class. E. g. such value that 99% of cells of that class have the higher certainty.

**Figure 1.** Example of the threshold identification for NK cells. It is the 1st percentile of correlations of NK cells with NK cells cluster centroid
![Thresholds example](images/thresholds_example.png)

Such an approach can be used to build a classifier based on correlations. Duan et al [2] use the correlations in the Dicriminative PCA space. We also implemented this approach.

# Running the code

1. Install dependencies:

```bash
$ pip install -r requirements.txt
```

Entrypoint for the code is the file [main.py](scripts/main.py) in the scripts directory. It has two modes:

## Training models
```
usage: main.py train [-h] -i -0 -P --models [...] - -n_genes --threshold --n_jobs
optional arguments:
-h, --help      show this help message and exit
-i, --input     Path to scRNA-seq dataset(s)
-o, --output    Path to the folder with results
-p , --path     Path to the folder with models
--models [...]  Names of the models to train. Available: svc, rforest, lgbm, corr
--n_genes       Number of genes to select for models training
--threshold     1 - if we want to use adaptive threshold, else 0
--n jobs        Number of jobs to run in parallel.
```

## Making predictions
```bash
usage: main. py predict [-h] -i -o -p --models [...] --n_genes --benchmarking --n_jobs
-h, --help      show this help message and exit
-i, --input     Path to scRNA-seq dataset(s)
-o, --output    Path to the folder with results
-p , --path     Path to the folder with models
--models [...]  Names of the models to train. Available: svc, rforest, lgbm, corr
--n_genes       Number of genes to select for models training
--benchmarking  0 if we want to run benchmarking, else 1
--n jobs        Number of jobs to run in parallel.
```

# References
1. Abdelaal, T., Michielsen, L., Cats, D. et al. A comparison of automatic cell identification methods for single-cell RNA sequencing data. Genome Biol 20, 194 (2019). https://doi.org/10.1186/s13059-019-1795-z
2. B. Duan, C. Zhu, G. Chuai, C. Tang, X. Chen, S. Chen, S. Fu, G. Li, Q. Liu, Learning for single-cell assignment. Sci. Adv. 6, eabd0855 (2020)
