# Meal Detection README

This repository contains Python code for meal detection and classification using machine learning. The project consists of two parts: training and testing.

## Table of Contents

- [Description](#description)
- [Prerequisites](#prerequisites)
- [Training Files](#training-files)
- [Test Files](#test-files)
- [Usage](#usage)
- [Functions](#functions)

## Description

This code is designed to analyze glucose and insulin data to detect and classify meals using a decision tree classifier. The project is divided into two main components: training and testing.

- **Training Files**: These files preprocess the training data, extract features from the data, train the decision tree classifier, and save the trained model. It performs meal and no-meal classification.

- **Test Files**: These files read the testing data, use the trained model to classify meals and no-meals, and save the results in a CSV file.

## Prerequisites

Before using the code, ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `datetime`
- `statistics`
- `scipy`
- `sklearn`
- `joblib`

You can install these libraries using pip:

```bash
pip install pandas numpy scipy scikit-learn joblib
```

## Training Files

- **Training Files**: This part of the code is responsible for training a decision tree classifier to classify meals and no-meals based on the provided training data.

## Test Files

- **Test Files**: This part of the code reads the testing data, uses the pre-trained decision tree classifier to classify meals and no-meals, and saves the results in a CSV file.

## Usage

To use the code, follow these steps:

1. **Training Files**:
   - Place the training data files in the specified locations.
   - Run the `main_function()` in the training code.
   - The code will preprocess the data, extract features, train the classifier, and save the trained model.

2. **Test Files**:
   - Prepare the testing data and save it in a CSV file.
   - Place the pre-trained model (`trained.pickle`) in the testing code directory.
   - Run the `main_function()` in the testing code.
   - The code will read the testing data, classify meals and no-meals, and save the results in a CSV file named `Result.csv`.

## Functions

- **Training Files**:
  - `get_meal_data(insulin, cmg, fileIndex)`: Extracts meal data from insulin and CGM data.
  - `get_no_meal_data(insulin, cmg, fileIndex)`: Extracts no-meal data from insulin and CGM data.
  - `meal_features(meals)`: Computes features from meal data.
  - `no_meal_features(noMeals)`: Computes features from no-meal data.
  - `main_function()`: Orchestrates the training process.

- **Test Files**:
  - `feature_matrix(test)`: Computes features from test data.
  - `main_function()`: Orchestrates the testing process.
