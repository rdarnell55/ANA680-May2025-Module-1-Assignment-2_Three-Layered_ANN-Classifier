# ANA680-May2025-Module-1-Assignment-2_Three-Layered_ANN-Classifier
# Bank Customer Churn Prediction using ANN

This project implements a deep learning solution to predict bank customer churn using an Artificial Neural Network (ANN). It follows a complete machine learning pipeline from data preprocessing to model training, evaluation, interpretability, and archiving.

## Dataset
- **File**: `Churn_Modelling.csv`
- **Target**: `Exited` (0 = stayed, 1 = churned)
- **Rows**: ~10,000
- **Features**: Demographic and financial attributes

## Features
- Label Encoding & One-Hot Encoding for categorical data
- Feature scaling with `StandardScaler`
- ANN with two hidden layers using `ReLU`, final layer using `Sigmoid`
- Early stopping to prevent overfitting
- SHAP explainability for feature importance analysis
- Visualizations of training progress and prediction behavior
- Output files archived into a `.zip` for reproducibility

## Model Architecture
- Input: 11 scaled features
- Hidden Layer 1: 6 neurons, ReLU
- Dropout: 0.3
- Hidden Layer 2: 6 neurons, ReLU
- Dropout: 0.3
- Output: 1 neuron, Sigmoid

## Evaluation
- Accuracy: **~85.6%**
- Confusion Matrix: **[[1502 93] [ 195 210]]**
- SHAP used for visualizing feature contributions

## Output Files
- `churn_predictions.csv`
- `model_weights.weights.h5`
- `training_history_log.csv`
- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`
- `churn_model_outputs.zip`

## How to Use
Clone the repo, ensure dependencies are installed (TensorFlow, scikit-learn, SHAP), and run the notebook or `.py` script. All outputs will be generated and archived automatically.
