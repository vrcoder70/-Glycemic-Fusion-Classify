# Meal Clustering and Classification

This repository contains Python code to process insulin and continuous glucose monitoring (CGM) data, extract features from the data, and perform clustering and classification of meals based on the extracted features.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Description](#description)
- [Usage](#usage)
- [Functions](#functions)

## Prerequisites

Before running the code, ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install numpy pandas scipy scikit-learn
```

## Description

The code provided here is designed to process insulin and CGM data, extract relevant features, and then cluster and classify meals based on those features. The key components of the code include:

1. **Data Preprocessing**: The code reads insulin and CGM data from CSV files, preprocesses the data, and aligns the timestamps to create a meal dataset.

2. **Feature Extraction**: It computes various features from the meal dataset, including velocity, acceleration, entropy, IQR, FFT coefficients, and power spectral density.

3. **K-Means Clustering**: The code performs K-Means clustering on the feature data to cluster meals into distinct categories. It calculates the Sum of Squared Errors (SSE), entropy, and purity for evaluation.

4. **DBSCAN Clustering**: The code also applies DBSCAN clustering to the feature data, using parameters like epsilon and minimum samples. It calculates the SSE, entropy, and purity for evaluation.

5. **Result Export**: The results, including SSE, entropy, and purity for both K-Means and DBSCAN, are saved to a CSV file named "Result.csv."

## Usage

To use the code, follow these steps:

1. Place your insulin data in a CSV file named "InsulinData.csv" and your CGM data in a CSV file named "CGMData.csv" in the `data` directory.

2. Run the `main_function()` by executing the code in your preferred Python environment.

3. The code will process the data, extract features, perform K-Means and DBSCAN clustering, and save the results in the "Result.csv" file.

## Functions

The code provides several functions for data processing, feature extraction, and clustering:

- `get_meal_data(insulin, cmg)`: Aligns insulin and CGM data to create a meal dataset, returning glucoseMatrix and ground truth labels.

- `get_features(meals)`: Extracts features from the meal data, including velocity, acceleration, entropy, IQR, FFT coefficients, and power spectral density.

- `kmean_clustering(data, ground_truth_labels)`: Performs K-Means clustering on the feature data, calculating SSE, entropy, and purity.

- `dbscan_clustering(data, ground_truth_labels)`: Applies DBSCAN clustering on the feature data with specified parameters, and calculates SSE, entropy, and purity.

- `main_function()`: The main function that orchestrates the entire process, from data loading to result export.
