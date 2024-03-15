# Biomarkers_Prediction

Project Title: Identifying Biomarkers driving Metastasis in Cancer

Objective:
Across several cancers, there poses a risk of metastasis. Identifying a small subset of genes in several cancers that spearhead the development of metastasis is the primary purpose of this project. Several cancer RNA-Sequencing and methylation datasets from The Cancer Genome Atlas (TCGA) including Bladder and Colorectal Cancer will be used. Microarray data from other sources such as CuMiDa would be used. Spearmans’ Correlation Correlation, Mutual Information and DESeq2 would be used to obtain a score and the magnitudes would be ranked in decreasing order to select top genes. For further filtering of genes, leave-one-out cross validation technique (Van’t Veer et al., 2002) will be used using Logistic Regression, Random Forest, ElasticNet Logistic Regression with ridge and lasso penalties incorporated. 
