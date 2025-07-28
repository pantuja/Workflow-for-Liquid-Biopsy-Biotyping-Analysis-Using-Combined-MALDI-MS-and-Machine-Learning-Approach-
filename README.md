# Workflow-for-Liquid-Biopsy-Biotyping-Analysis-Using-Combined-MALDI-MS-and-Machine-Learning-Approach-
An open-source R script for end-to-end MALDI-MS data processing and machine learning-based classification in liquid biopsy studies.
This repository contains an open-source and user-modifiable R script used in the article:

**"End-to-End Workflows for Liquid Biopsy Biotyping Analysis Using Combined MALDI MS and Machine Learning Approach"**  
*by Lukáš Pečinka, Jaromíra Pantučková, et al.*

DOI: 

---

## Purpose

The R script provided in this repository implements a transparent, reproducible and fully customizable workflow for the preprocessing, analysis, and classification of MALDI-TOF mass spectrometry data, specifically tailored for liquid biopsy studies in clinical research.

---

## Workflow Overview

The code is divided into several modular parts:

1. **Preprocessing of raw spectra**
   - Trimming, smoothing, transformation, baseline correction, normalization
   - Alignment and averaging of technical replicates
   - Peak detection, binning, filtering, and feature matrix construction

2. **Exploratory data analysis**
   - Principal Component Analysis (PCA)
   - PLS-DA, OPLS-DA, and visualization of discriminatory features

3. **Supervised machine learning models**
   - PLS-DA, Random Forest, Support Vector Machine (SVM), Artificial Neural Networks (ANN)
   - Multiple validation strategies (LOOCV, repeated CV, train/test split)
   - Evaluation metrics: accuracy, sensitivity, specificity, AUC

4. **Model validation and feature selection**
   - Permutation testing for overfitting
   - Wilcoxon-based feature filtering

---

## Requirements

- R (version ≥ 4.3)
- R packages:
  - `MALDIquant`, `MALDIquantForeign`, `MALDIrppa`, `ropls`, `caret`, `ggplot2`, `readxl`, `openxlsx`, `nnet`, etc.

Install required packages in R using:

```R
install.packages(c("MALDIquant", "MALDIquantForeign", "MALDIrppa", "ggplot2", "caret", "nnet", "openxlsx", "readxl"))
```

---

## Input Format

- Input files should be in `mzML` format and organized in folders per experimental group (e.g., `HD_1/`, `MM_1/`).
- Each sample should have multiple technical replicates (e.g., `Subject01_1.mzML`, `Subject01_2.mzML`, ...).
- Filenames are used as spectrum identifiers.

---

## How to Use

1. Adjust `setwd()` and `group_dirs` in the script to match your data structure.
2. Follow script annotations to run each section step by step.
3. Output of each step is saved as `.rds` for easy reproducibility.
4. Visualizations and results are automatically saved as `.pdf` and `.xlsx`.

---

## Contact

Author of the script: **Jaromíra Pantučková**  
534266@mail.muni.cz

---

## Citation

If you use this workflow, please cite the original article and link to this GitHub repository.

