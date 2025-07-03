# EL_task_7
This task demonstrates how to use Support Vector Machines (SVM) for binary classification of breast cancer tumors (malignant vs. benign) using the breast-cancer.csv dataset

## Overview

## 1. Data Loading and Exploration
Loads the dataset using pandas.
Displays info, statistics, first few rows, and checks for missing values.
## 2. Data Cleaning and Preparation
Drops the id column (not useful for prediction).
Encodes the diagnosis column: M → 1 (malignant), B → 0 (benign).
Splits the data into features (X) and target (y).
Further splits data into training and test sets (80/20 split).
## 3. Feature Scaling
Applies standardization to features using StandardScaler to ensure all features contribute equally to the SVM.
## 4. Model Training and Evaluation
Trains two SVM classifiers:
Linear kernel
RBF (Radial Basis Function) kernel
Evaluates both models on the test set and prints accuracy scores.
## 5. Decision Boundary Visualization
Selects two features (radius_mean, texture_mean) for 2D visualization.
Trains SVMs (linear and RBF) on these features.
Plots decision boundaries and data points to visually compare how each kernel separates the classes.
## 6. Hyperparameter Tuning
Uses GridSearchCV to find the best C and gamma values for the RBF SVM.
Performs 5-fold cross-validation during the search.
Prints the best hyperparameters and corresponding accuracy.
## 7. Cross-Validation
Evaluates the best RBF SVM using 5-fold cross-validation on the training set.
Prints the mean accuracy and standard deviation.
## Results
 ## Accuracy scores for both linear and RBF SVMs are printed. 
         "0.956140350877193 & 0.9824561403508771"
 ## Best hyperparameters for RBF SVM are displayed.
          Best parameters  values: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
## Cross-validation scores show the robustness of the best model.
          Best cross validation: 0.9736263736263737
          Cross-validation scores: [0.98901099 0.96703297 0.98901099 0.97802198 0.93406593]
         Mean Accuracy: 0.9714285714285715
         Standard Deviation: 0.020381579110979577
Decision boundary plots provide visual insight into model behavior.
