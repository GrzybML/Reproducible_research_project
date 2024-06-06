# Accident Detection Analysis

This repository contains the code and data for accident detection analysis using various machine learning models. The primary goal of this project is to replicate the results and findings of a specific research paper that investigates the impact of neighboring sensor measurements and Time-to-Detection Accuracy (TTDA) on accident detection.

## Table of Contents
- [Introduction](#introduction)
- [Models Used](#models-used)
- [Metrics and Evaluation](#metrics-and-evaluation)
- [Usage](#usage)
- [Branch Structure](#branch-structure)
- [File Descriptions](#file-descriptions)
- [Contributing](#contrubiting)

## Members

- Pola Parol
- Natalia Roszczypała
- Olek Wieliński
- Michał Grzyb

## Introduction

In this project, we conduct accident detection analysis by utilizing the data from various sensors placed at different distances from the accident location. The analysis is performed using three different classifiers: Logistic Regression, Random Forest, and XGBoost. We compare the classification results under two settings:

1. **Setting 1:** Using up to five neighboring sensors located upstream from the accident.
2. **Setting 2:** Using symmetric sensors located both upstream and downstream from the accident location (up to five sensors in each direction).

We hypothesize that accidents significantly affect upstream and downstream traffic up to a certain distance, and using these features helps to design better classifiers. The analysis includes varying the Time-to-Detection Accuracy (TTDA) from 0 minutes to 7 minutes at 30-second intervals.

## Models Used

We use the following machine learning models for the analysis:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

## Metrics and Evaluation

The performance of the models is evaluated using the following metrics:
- **Detection Rate (DR)**
- **False Alarm Rate (FAR)**
- **AUC-ROC**
- **AUC-PR**

## Results
The final results present the **heatmaps** generated in the separate folder "output". Heatmaps are visual representations that display the performance metrics of the models across different settings. They help in identifying patterns and insights by showing variations in model performance based on different parameters, such as the number of sensors and time-to-detection accuracy.

In addition to heatmaps, the notebook also generates **SHAP values** for the selected model. SHAP values provide interpretability by showing the contribution of each feature to the model's predictions. **Summary statistics** for the models and settings are also generated, giving a comprehensive overview of the model performances under different configurations.

## Usage
To run the analysis, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/GrzybML/Reproducible_research_project
   cd Reproducible_research_project

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the main analysis script:
   ```bash
   jupyter nbconvert --to notebook --execute I24_main.ipynb

## Branch Descriptions

This repository has three branches:

**main**: The main branch contains the core structure and essential files.
**I24_data**: Contains the analysis and results for the I-24 intersection.
**I75_data**: Contains the analysis and results for the I-75 intersection.

## File Descriptions

**I24_main.ipynb** and **I75_main.ipynb**
Contains the Jupyter notebook for the analysis conducted on the I-24/I-75 intersection. It includes data loading, preprocessing, model training, and evaluation.

Sections:
- Data Loading: Loads the data for I-24/I-75 intersection.
- Data Preprocessing: Cleans and prepares the data for analysis.
- Model Training and Evaluation: Trains the models and evaluates their performance.
- Sensitivity Analysis: Conducts sensitivity analysis on the models.
- Results Visualization: Generates heatmaps and SHAP plots for interpreting model performance.

**data_preparation.py**
Contains functions for data loading, cleaning, and feature generation.

**ML_classifier.py**
Contains the machine learning classifier class and related functions.

Classes and Functions:
- MLClassifier: A class that encapsulates model training, evaluation, and sensitivity analysis.
- __init__(self, data, target, output_dir='output'): Initializes the MLClassifier with data, target variable, and output directory.
- preprocess_data(self, data, drop_time_diff=True): Prepares the data for model training by cleaning and splitting into training and testing sets.
- train_models(self): Trains the models using SMOTE and GridSearchCV.
- sensitivity_analysis(self): Performs sensitivity analysis on different time differences and sensor hops.
- train_models_specific_feature(self, X_train, X_test, y_train, y_test, feature_name): Trains models on specific features and evaluates performance.
- generate_heatmap(self, results): Generates heatmaps for visualizing performance metrics across different settings.
- generate_summary_table(self, results): Creates a summary table of the performance metrics.
- plot_shap_values(self, results): Plots SHAP values to interpret model predictions.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

