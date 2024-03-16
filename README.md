# Biomarkers_Prediction

Project Title: Identifying Biomarkers driving Metastasis in Cancer

Objective:
Across several cancers, there poses a risk of metastasis. Identifying a small subset of genes in several cancers that spearhead the development of metastasis is the primary purpose of this project. Several cancer RNA-Sequencing and methylation datasets from The Cancer Genome Atlas (TCGA) including Bladder and Colorectal Cancer will be used. Microarray data from other sources such as CuMiDa would be used. Spearmans’ Correlation Correlation, Mutual Information and DESeq2 would be used to obtain a score and the magnitudes would be ranked in decreasing order to select top genes. For further filtering of genes, leave-one-out cross validation technique (Van’t Veer et al., 2002) will be used using Logistic Regression, Random Forest, ElasticNet Logistic Regression with ridge and lasso penalties incorporated. 

Python Notebooks:
1. colorectalCancerCuMiDaMicroarrayDataSet.py is code for colorectal cancer dataset from CuMiDa. Correlation coefficient ranking is carried out followed by leave-one-out cross-validation using Random Forest.
2. bladderCancer(MI+CC).py is code for bladder cancer dataset from TCGA. Both mutual information and correlation coefficient scores were calculated. Common top 2000 genes from both methods are used for leave-one-out cross-validation using Random Forest and Logistic Regression.
