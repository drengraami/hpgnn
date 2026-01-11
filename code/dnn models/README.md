# Baseline DNN models for aCDOM(440) / Chla / Secchi_depth / TSS / Turbidity Estimation

## Description

This subdirectory includes codes for purely data-driven Deep Neural Network (DNN) models used as a baseline model for estimating Absorption of Colored Dissolved Organic Matter at 440 nm, Chlorophyll-a, Secchi Disk Depth, Total Suspended Solids, and Turbidity from Sentinel-2 reflectance data. Unlike the HPGNN models, these models do not incorporate any other constraint and relies solely on Mean Squared Error (MSE).

*  Input Data and Preprocessing

* Input features: Sentinel-2 reflectance bands: B1, B2, B3, B4, B5, B6, B7, B8

* Target variable: aCDOM440 / Chla / Secchi_depth / TSS / Turbidity

* Data source: simulated_data_training_validation and matchup_data_test excel files given in data subdirectory

* Preprocessing Steps:

    * Outlier removal: Quantile-based filtering (10th–90th percentile) applied to the target variable

    * Normalization: Min–Max scaling of reflectance bands

    * Data split: 80% training / 20% testing

## Model Architecture

* Fully connected Deep Neural Network

* 2–6 hidden layers (optimized via Bayesian hyperparameter tuning)

* Hidden units per layer: 32–1024

* Activation function: ReLU

* Regularization:

    * Batch Normalization

    * Dropout (0.2–0.5)

* Output layer: Single neuron predicting target variable

## Hyperparameter Optimization

* Optimization method: Bayesian Optimization (Keras-Tuner)

* Tuned parameters:

    * Number of hidden layers

    * Neurons per layer

    * Dropout rates

    * Learning rate

* Objective: Minimize validation MSE

## Model Training

* Optimizer: Adam

* Loss function: MSE

* Batch size: 64

* Maximum epochs: 1000

* Callbacks:

    * Early stopping

    * Learning rate reduction on plateau

## Model Evaluation

* Model performance is evaluated using:

    * R² (Coefficient of Determination)

    * RMSE (Root Mean Squared Error)

    * MAE (Mean Absolute Error)

* Evaluation is performed on:

    * Training dataset (80%)

    * Validation dataset (20%)

    * Matchup test dataset

## Requirements

* Python

* tensorflow, keras, keras-tuner, pandas, numpy, scikit-learn



