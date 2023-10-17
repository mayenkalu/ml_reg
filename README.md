# Regression in Machine Learning

This repository provides a comprehensive understanding of various regression techniques, evaluation metrics, essential preprocessing steps, and model tuning techniques in Machine Learning.

## Table of Contents

1. [Simple and Multiple Linear Regression](#simple-and-multiple-linear-regression)
2. [Non-linear Regression](#non-linear-regression)
3. [Regression Metrics](#regression-metrics)
4. [Penalization Methods](#penalization-methods)
5. [Model Tuning](#model-tuning)
6. [Data Splitting](#data-splitting)
7. [Resampling](#resampling)
8. [Cross-Validation](#cross-validation)

### Simple and Multiple Linear Regression

- **Simple Linear Regression**: Models the relationship between a single independent variable and the dependent variable using a linear equation.
- **Multiple Linear Regression**: Expands on simple linear regression to include multiple independent variables.

### Non-linear Regression

- An approach to model the non-linear relationships between independent and dependent variables.
- Common methods include polynomial, logarithmic, and exponential regression.

### Regression Metrics

- **Mean Absolute Error (MAE)**: Average of absolute differences between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Square root of the average squared differences between predicted and actual values.
- **R-squared**: Represents the proportion of variance in the dependent variable that's explained by independent variables.
- **Bias**: The difference between the expected prediction of our model and the correct value.
- **Variance**: The variability of model predictions for different training sets.

### Penalization Methods

- Techniques to prevent overfitting by adding a penalty to the loss function.
- **Ridge Regression (L2 regularization)**: Penalizes the squared magnitude of coefficients.
- **Lasso Regression (L1 regularization)**: Penalizes the absolute magnitude of coefficients, often leading to feature selection.
- **Elastic Net**: Combines L1 and L2 penalties of the Lasso and Ridge methods.

### Model Tuning

- **Model Parameters**: Internal characteristics of the model that are learned from the data.
- **Hyperparameters**: Settings on the model which are externally adjusted to improve performance.
  - **Grid Search**: Exhaustive search through a manually specified subset of hyperparameters.
  - **Random Search**: Randomly selects combinations of hyperparameters to find the best solution.

### Data Splitting

- Dividing data into subsets, commonly training and test sets, to train and evaluate models.
- Helps in assessing the model's performance on unseen data.
- In common practice, the dataset is divided into training, validation or dev and testing sets.

### Resampling

- Techniques to estimate model accuracy using subsets of data.
- Repeatedly selecting samples from original dataset using the samples to obtain information about model.

### Cross-Validation

- Partitioning the original sample into a training set to train the model and a test set to evaluate it.
- **k-Fold Cross-Validation**: Dividing data into 'k' subsets. The model is trained on k-1 folds and tested on the remaining fold. This process is repeated k times.
- Used to generalise and prevent overfitting in machine learning.
