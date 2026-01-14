# Obesity Level Prediction: ML Algorithms from Scratch
Implementation of fundamental Machine Learning algorithms (Regression, Decision Trees, KNN, Neural Networks) from scratch using only NumPy, applied to obesity level prediction.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ClaudiaCornacchia/ML_Project/blob/main/Obesity_Level_Prediction_ML_Project.ipynb)


**Author:** Claudia Cornacchia   
**Context:** Foundations of AI Course Project (from my Bachelor's Degree)

## Overview
This project focuses on the implementation of fundamental Machine Learning algorithms **from scratch** (without using `scikit-learn` for model training) to predict obesity levels based on eating habits and physical condition.
The goal was to demonstrate a deep understanding of the mathematical foundations behind these algorithms by implementing the optimization logic, gradient descent, and backpropagation manually.

## Algorithms Implemented
I implemented the following models using only `NumPy` and `Pandas`:

### 1. Linear Regression (Weight Prediction)
- **Method:** Closed-form solution (Normal Equation) using matrix operations.
- **Optimization:** Calculated coefficients $c = (X^TX)^{-1}X^TY$ to minimize the Mean Squared Error.

### 2. Decision Tree (Binary Classification)
- **Method:** Recursive greedy approach.
- **Metric:** Information Gain and Entropy to select the best split at each node.
- **Hyperparameter:** Tuned `max_depth` to prevent overfitting.

### 3. Logistic Regression (Binary Classification)
- **Method:** Gradient Descent optimization.
- **Loss Function:** Binary Cross-Entropy (Log Loss).
- **Regularization:** Implemented **L2 Ridge Regularization** to reduce model complexity.

### 4. K-Nearest Neighbors (KNN)
- **Method:** Vectorized distance calculation for efficiency.
- **Distance Metrics:** Comparison between **Euclidean** and **Manhattan** distances.

### 5. Neural Network (Regression)
- **Architecture:** Feed-forward network with variable hidden layers.
- **Activation:** **ReLU** for hidden layers.
- **Optimization:** Mini-Batch Gradient Descent with Backpropagation.
- **Initialization:** He/Normal initialization to prevent vanishing gradients.

## 📊 Results & Performance
The models were evaluated using a Grid Search for hyperparameter tuning.

| Algorithm | Task | Metric | Value |
|:---:|:---:|:---:|:---:|
| **Linear Regression** | Regression | RMSE | **4.92** |
| **Neural Network** | Regression | RMSE | **4.65** |
| **Decision Tree** | Classification | Accuracy | **0.96** |
| **KNN** | Classification | Accuracy | **0.95** |
| **Logistic Regression** | Classification | Accuracy | **0.78** |

*Note: The Neural Network outperformed Linear Regression due to its ability to capture non-linear relationships, while the Decision Tree achieved the highest classification accuracy.*

## 🛠️ Technologies
- **Python** (NumPy, Pandas)
- **Matplotlib / Seaborn** (Data Visualization)
- **Scikit-learn** (Used ONLY for metric verification and data splitting)

## 💻 How to Run
1. Click the **"Open in Colab"** badge above.
2. The notebook automatically handles data downloading and preprocessing.
