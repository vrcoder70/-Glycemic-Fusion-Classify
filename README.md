# Glucose Analysis, Meal Detection, and Meal Clustering and Classification

This repository combines three different projects related to glucose analysis, meal detection, and meal clustering and classification.

## Table of Contents

- [Glucose Analysis](#glucose-analysis)
- [Meal Detection](#meal-detection)
- [Meal Clustering and Classification](#meal-clustering-and-classification)
- [Prerequisites](#prerequisites)
- [Usage](#usage)

## Glucose Analysis

### Description

The **Glucose Analysis** project provides Python code for analyzing glucose data, including continuous glucose monitoring (CGM) and insulin data. It calculates various statistics related to glucose levels, such as hyperglycemia and hypoglycemia. The code segments the data into different timeframes (whole day, daytime, nighttime) and calculates statistics for each segment.

### Prerequisites

Before using the code for glucose analysis, make sure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `datetime`
- `statistics`

You can install these libraries using pip:

```bash
pip install numpy pandas
```

### Usage

To use the Glucose Analysis code:

1. Place your CGM data in a CSV file named `CGMData.csv` and insulin data in a CSV file named `InsulinData.csv`. Ensure these data files are in the `data` directory.

2. Run the `main_function()` to process the CGM and insulin data, calculate glucose level statistics, and save the results in a CSV file named `Result.csv`.

## Meal Detection

### Description

The **Meal Detection** project focuses on detecting and classifying meals using machine learning. It consists of two main parts: training and testing.

- **Training Files**: These files preprocess the training data, extract features, train a decision tree classifier, and save the trained model for meal and no-meal classification.

- **Test Files**: These files read the testing data, use the pre-trained decision tree classifier to classify meals and no-meals, and save the results in a CSV file.

### Prerequisites

Before using the Meal Detection code, make sure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `joblib`

You can install these libraries using pip:

```bash
pip install pandas numpy scipy scikit-learn joblib
```

### Usage

To use the Meal Detection code:

1. **Training Files**:
   - Place the training data files in the specified locations.
   - Run the `main_function()` in the training code.
   - The code will preprocess the data, extract features, train the classifier, and save the trained model in a file named `trained.pickle`.

2. **Test Files**:
   - Prepare the testing data and save it in a CSV file.
   - Place the pre-trained model (`trained.pickle`) in the testing code directory.
   - Run the `main_function()` in the testing code.
   - The code will read the testing data, classify meals and no-meals, and save the results in a CSV file named `Result.csv`.

## Meal Clustering and Classification 

### Description

The **Meal Clustering and Classification** project processes insulin and continuous glucose monitoring (CGM) data, extracts features, and performs clustering and classification of meals based on the extracted features.

The code includes data preprocessing, feature extraction, and the application of K-Means and DBSCAN clustering algorithms to classify meals into distinct categories. It calculates various metrics for evaluation, such as Sum of Squared Errors (SSE), entropy, and purity.

### Prerequisites

Before using the Meal Clustering and Classification code, make sure you have the following libraries installed:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install numpy pandas scipy scikit-learn
```

### Usage

To use the Meal Clustering and Classification code:

1. Place your insulin data in a CSV file named `InsulinData.csv` and your CGM data in a CSV file named `CGMData.csv`.

2. Run the `main_function()` by executing the code in your preferred Python environment.

3. The code will process the data, extract features, perform K-Means and DBSCAN clustering, and save the results in a CSV file named `Result.csv`.

These three projects, combined in one repository, provide comprehensive functionality for glucose data analysis, meal detection, and meal clustering and classification.

