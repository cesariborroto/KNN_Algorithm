# K-Nearest Neighbors (KNN) Algorithm

This repository contains a Jupyter Notebook file implementing the K-Nearest Neighbors (KNN) algorithm for classification. The algorithm is applied to the Iris dataset, which is a popular dataset in machine learning.

## Prerequisites
- Python 3.x
- Jupyter Notebook
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

## Dataset
The Iris dataset is used in this notebook, which is included in the `scikit-learn` library. The dataset consists of measurements of four features of three different species of Iris flowers. The goal is to classify the species based on the feature measurements.

## Algorithm Overview
The KNN algorithm is a simple yet effective classification algorithm. It classifies new data points based on their similarity to known data points in the training set. The algorithm works by calculating the distance between the new data point and all other data points in the training set. It then selects the K nearest neighbors and assigns the class label based on majority voting.

## Notebook Contents
1. Importing Required Libraries: The necessary libraries, such as `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`, are imported.
2. Loading the Dataset: The Iris dataset is loaded from the `scikit-learn` library.
3. Data Preprocessing: The dataset is divided into training and testing sets using the `train_test_split` function from `scikit-learn`.
4. Exploratory Data Analysis: Basic exploration of the dataset is performed using `pandas` and `matplotlib`. The top 5 rows of the dataset are displayed, and the length and shape of the data are printed.
5. KNN Algorithm Implementation: The KNN algorithm is implemented using the `KNeighborsClassifier` class from `scikit-learn`. The algorithm is trained on the training set and tested on the testing set.
6. Model Evaluation: The accuracy of the KNN algorithm is evaluated using the `metrics.accuracy_score` function from `scikit-learn`.
