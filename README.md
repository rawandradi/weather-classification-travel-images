# Weather Classification from Travel Images

A machine learning project for classifying weather conditions from travel photographs.
## Overview

Given a travel image, the goal is to classify the weather condition into one of five classes: **Sunny, Cloudy, Rainy, Snowy, or Not Clear**.

Three ML models were implemented and compared:
- K-Nearest Neighbors (KNN) — baseline
- Random Forest Classifier
- Support Vector Machine (SVM) with RBF kernel

## Dataset

- 1,457 raw samples collected from multiple CSV sources
- After cleaning: **959 usable samples** with 8 categorical attributes
- Strong class imbalance: Sunny dominates at 57.66%

## Feature Extraction

| Feature | Description | Dimensions |
|---|---|---|
| HSV Color Histogram | Hue, Saturation, Value channels (16 bins each) | 48-dim |
| SIFT Bag-of-Words | Visual vocabulary with MiniBatch K-Means (k=200) | 200-dim |
| **Final vector** | HSV + SIFT concatenated | **248-dim** |

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| KNN (k=1) | 47.5% | 0.33 |
| KNN (k=3) | 53.8% | 0.34 |
| SVM (RBF, C=1) | 56.4% | 0.37 |
| **Random Forest** | **66.7%** | **0.51** |

Random Forest was selected as the best model.

## Project Structure
├── DataProcessing.ipynb         # Data merging, cleaning, preprocessing
├── EDA.py                       # Exploratory data analysis & visualizations
├── KNN_Model.ipynb              # KNN baseline implementation
├── RF_Models.ipynb              # Random Forest with hyperparameter tuning
├── weather_svm_model.py         # SVM classifier implementation
├── extract_hsv.py               # HSV feature extraction
├── extract_sift.py              # SIFT feature extraction
├── build_final_features.py      # Feature vector construction
├── download_images.py           # Image downloading pipeline
├── filter_real_images.py        # Image validation and filtering
├── finalize_dataset.py          # Final dataset preparation
├── dataset_clean.csv            # Cleaned dataset
└── Report.pdf                   # Full project report

## Key Findings

- Class imbalance (23:1 ratio) was the primary challenge across all models
- Most misclassifications occurred when Cloudy/Snowy/Rainy samples were predicted as Sunny
- Random Forest outperformed SVM and KNN due to its ability to capture non-linear feature relationships

## Authors

- Rawand Radi — [rawandradi@gmail.com](mailto:rawandradi@gmail.com)
- Tasneem Shelleh
