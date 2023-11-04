# Glucose Analysis README

This repository contains Python code for analyzing glucose data to calculate statistics related to glucose levels. It processes continuous glucose monitoring (CGM) and insulin data, calculates various statistics, and stores the results in a CSV file. This README provides an overview of the code, its functionality, and how to use it.

## Table of Contents

- [Description](#description)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Functions](#functions)

## Description

The code provided here is designed to analyze glucose data and calculate statistics for different glucose level ranges, including hyperglycemia, hypoglycemia, and more. It operates on both CGM and insulin data and generates statistics for the whole day, daytime, and nighttime.

## Prerequisites

Before using the code, ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `datetime`
- `statistics`

You can install these libraries using pip:

```bash
pip install numpy pandas
```

## Getting Started

1. **Data Preparation**: Place your CGM data in a CSV file named `CGMData.csv` and insulin data in a CSV file named `InsulinData.csv`. Make sure the data files are in the `data` directory.

2. **Code Overview**: The code reads and processes the data, interpolates missing values, and calculates statistics based on glucose levels.

3. **Result File**: The calculated statistics are saved in a CSV file named `Result.csv`.

## Usage

To use the code, follow these steps:

1. Run the `main_function()` by executing the code in your preferred Python environment.

2. The code will process the CGM and insulin data, calculate statistics for different glucose level ranges, and store the results in the `Result.csv` file.

3. The results include statistics for the whole day, daytime, and nighttime, for various glucose level ranges.

## Functions

The code consists of several functions:

- `calculate_statistics(dictionary)`: Calculates statistics for hyperglycemia, hypoglycemia, and other glucose level ranges based on the provided dictionary.

- `create_segments(cmgData)`: Segments the CGM data into day, daytime, and nighttime segments and calculates statistics for each segment.

- `main_function()`: The main function that orchestrates the entire process, from data loading to result export.

