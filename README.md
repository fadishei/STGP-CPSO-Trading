# Introduction

This is the source code developed for the experiments of the framework proposed in this paper for extracting trading rules from stock market data:

Faezeh Kavi and Hamid Fadishei, "Extracting trading rules from stock market data using evolutionary computation and nested parameter optimization".

This framework utilizes a specialized version of the STGP algorithm, which is equipped with a nested PSO algorithm for constant optimization.

# Structure

This project directory contains three subdirectories:

- Dataset
- Experiments
- Trend-Identification

## Dataset

The code in this directory reads all S&P 500 stocks from Wikipedia and downloads their historical data from Yahoo! Finance using the yfinance library. The data includes open, high, low, and close prices.

## Trend-Identification

The code in this directory applies linear regression to different stock close prices in order to categorize them according to their trend in the market which can be either upterend, or downtrend, or sideway.

## Experiments

This is the implementation of our proposed framework and the related experiments.

# Requirements

This source code is implemented in Python language as a number of Jupyter Notebooks. The required dependencies can be found in requirements.txt file. Depending on the operating system environment, some libraries packages may need to be installed prior to installing the pypthon requirements. For example, on an Ubuntu version 22.04, the following commands may be used:

    sudo apt install graphviz-dev
	pip3 install -r requirements.txt

# Usage

Initially, you need to download S&P 500 historical data using the Dataset subproject. Then, you can categorize stocks using Stock Trend.

In the Experiments folder, there are some files:

- ### util.py

This file contains some utility functions used in the code.

- ### stgp_cpso.py

This is the main implementation of the proposed framework.

- ### stgp.py

This file is a STGP implementation developed for the purpose of comparing the results

- ### run.ipynb

This file serves as the entry point for testing our proposed framework using three categories with random parameters and PSO-generated parameters. You can place your stocks in three arrays based on their trends. After running this code, it will generate output for all the specified categories.