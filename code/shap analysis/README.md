# SHAP analysis for HPGNN models developed for aCDOM(440) / Chla / Secchi_depth / TSS / Turbidity Estimation

## Description

This subdirectory contains script for interpreting HPGNN models' predictions of water quality parameters using SHAP (SHapley Additive exPlanations).

## Purpose

* Quantify feature importance of Sentinel-2 spectral bands on water quality targets: aCDOM440, Chla, Secchi_depth, TSS, and Turbidity

* Visualize global and local SHAP contributions.

* Identify positive or negative impacts of spectral bands on predictions.

## Workflow

1) Load and preprocess data (simulated_data_training_validation and matchup_data_test excel files given in data subdirectory)

    * Remove outliers using IQR method.

    * Scale spectral bands (B1–B8) using MinMaxScaler.

2) Load pretrained HPGNN models

    * Load models and corresponding hyperparameters (learning rate, adaptive weights w1–w6).

3) Compute SHAP values

    * Use DeepExplainer with background samples from training data.

    * Generate summary plots, mean SHAP bar plots, and local SHAP contributions for selected percentiles (P10, P50, P90).

4) Save plots

    * Global summary: global_shap_beeswarm__<target>_HPGNN.png

    * Global feature importance: global_shap_bar_<target>_HPGNN.png

    * Local sample contributions: local_shap_bar_<target>_HPGNN.png

5) Print summary statistics

    * Mean and std of SHAP values.

    * Feature ranking and directionality of effects.

## Requirements

* Python

* tensorflow, keras, shap, pandas, numpy, matplotlib, seaborn, scikit-learn

## Output

* SHAP plots (global and local)

* Feature importance ranking printed in console



