# Ablation study models for aCDOM(440) / Chla / Secchi_depth / TSS / Turbidity Estimation

## Description

This subdirectory includes codes to perform systematic ablation study within the HPGNN framework for Absorption of Colored Dissolved Organic Matter at 440 nm, Chlorophyll-a, Secchi Disk Depth, Total Suspended Solids, and Turbidity retrieval from Sentinel-2 reflectance data. The objective is to quantify the individual and combined contributions of physics-guided and statistical regularization constraints, while keeping the tuned HPGNN architecture and hyperparameters fixed.

## Data and Preprocessing

* Input features: Sentinel-2 reflectance bands: B1, B2, B3, B4, B5, B6, B7, B8

* Target variable: aCDOM440 / Chla / Secchi_depth / TSS / Turbidity

* Data source: simulated_data_training_validation and matchup_data_test excel files given in data subdirectory

* Preprocessing

    * Outlier removal: Quantile-based filtering (10th–90th percentile) on the target variable

    * Normalization: Min–Max scaling of reflectance bands

    * Data split: 80% training / 20% validation

## HPGNN Architecture (Fixed)

* All ablation experiments use the same HPGNN architecture, defined by prior hyperparameter tuning:

    * Fully connected neural backbone

    * Multiple hidden layers with tuned depth

    * Tuned number of neurons per layer

    * ReLU activation

    * Batch Normalization

    * Dropout regularization

    * Single regression output for target variable

* No changes are made to the network structure during ablation experiments.

## Hyperparameter Usage (Pre-tuned)

* Hyperparameters were optimized beforehand using Bayesian Optimization

* The best-performing HPGNN configuration is reused for all ablation runs, including:

    * Number of layers

    * Neurons per layer

    * Dropout rates

    * Learning rate

    * Batch size

* This ensures that performance differences arise solely from loss-function modifications, not from re-optimization.

## Hybrid Physics-Guided Loss Function

* The baseline HPGNN model integrates data-fidelity with physics-guided and statistical regularization constraints through a composite loss function:

    * Mean Squared Error (MSE)
    * Non-negativity Constraint
    * Empirical Bio-Optical Constraint
    * Uncertainty Loss
    * Variance Loss
    * Smoothness Loss
    * Gradient Regularization

## Ablation Configurations

* Ablation experiments are conducted by selectively removing loss components, while all other settings remain unchanged:

    * No Non-negativity Constraint

    * No Empirical Bio-Optical Constraint

    * No Uncertainty Regularization

    * No Variance Regularization

    * No Smoothness Constraint

    * No Gradient Regularization

    * No Physics-Guided Constraints
        (non-negativity + Empirical Bio-Optical Constraint + gradient)

    * No Statistical Regularization Constraints
        (uncertainty + variance + smoothness)

* Each configuration represents a controlled modification of the baseline HPGNN.

## Training Strategy (Identical Across Ablations)

* Optimizer: Adam (tuned learning rate)

* Batch size: 64

* Maximum epochs: 1000

* Callbacks:

    * Early stopping

    * Learning rate reduction

    * Adaptive loss-weight updating

## Evaluation Metrics

* Model performance is evaluated on:

    * Training dataset (80%)

    * Validation dataset (20%)

    * Matchup test dataset

* Using:

    * R²

    * RMSE

    * MAE

## Model Outputs

* Trained model (.h5)

* Results are summarized to enable direct quantitative comparison across ablation configurations.

## Requirements

* Python

* tensorflow, keras, keras-tuner, pandas, numpy, scikit-learn, os
