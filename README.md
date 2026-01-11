# Explainable Hybrid Physics-Guided Neural Network (HPGNN) for Diverse Inland Lakes Water Quality Inversion

## Introduction 

This repository contains the code, data, and models for modeling and generating water quality products (Absorption of Colored Dissolved Organic Matter at 440 nm (aCDOM(440)), Chlorophyll-a, Secchi Disk Depth, Total Suspended Solids, and Turbidity) from Sentinel-2 imagery using Hybrid Physics-Guided Neural Network (HPGNN).

The repository is organized into three main directories:

code/ – Python and JavaScript scripts for model development, ablation studies, data downloading, product generation, and SHAP analysis and README files for brief explanation of each subdirectory.

data/ – Contains training, validation, and testing datasets (Excel files) and example Sentinel-2 image with cloud mask (TIF files) and README file for brief explanation of data.

models/ – Pre-trained models and associated hyperparameters and scalers for generating water quality products and README file for brief explanation of models.

## Repository Structure

```

hpgnn/
│
├── code/
│   ├── ablation study/                     # Ablation study scripts for each water quality parameter
│   │   ├── ablation_study_acdom.py
│   │   ├── ablation_study_chla.py
│   │   ├── ablation_study_sdd.py
│   │   ├── ablation_study_tss.py
│   │   ├── ablation_study_turbidity.py
│   │   └── README.md                       # Provides brief explanation about ablation study
│   │
│   ├── dnn models/                         # Scripts to train and tune DNN models for each parameter
│   │   ├── sentinel_dnn_acdom_tuner.py
│   │   ├── sentinel_dnn_chla_tuner.py
│   │   ├── sentinel_dnn_sdd_tuner.py
│   │   ├── sentinel_dnn_tss_tuner.py
│   │   ├── sentinel_dnn_turbidity_tuner.py
│   │   └── README.md                       # Provides brief explanation about DNN models
│   │
│   ├── downloading sentinel-2/             # Scripts to download Sentinel-2 mosaic and surface reflectance
│   │   ├── download_sentinel-2_mosaic.js
│   │   ├── download_sentinel-2_surfacereflectance_matchup.js
│   │   └── README.md                       # Provides brief explanation about downloading Sentinel-2 mosaic and surface reflectance
│   │
│   ├── hpgnn models/                       # Scripts to train and tune HPGNN models for each parameter
│   │   ├── sentinel_hpgnn_acdom_tuner.py
│   │   ├── sentinel_hpgnn_chla_tuner.py
│   │   ├── sentinel_hpgnn_sdd_tuner.py
│   │   ├── sentinel_hpgnn_tss_tuner.py
│   │   ├── sentinel_hpgnn_turbidity_tuner.py
│   │   └── README.md                       # Provides brief explanation about HPGNN models
│   │
│   ├── product generation/                 # Generate water quality products from Sentinel-2 using pre-trained models
│   │   ├── generate_products.py
│   │   └── README.md                       # Provides brief explanation about water quality product generation
│   │
│   └── shap analysis/                      # Scripts for SHAP explainability analysis
│       ├── shap_analysis.py
│       └── README.md                       # Provides brief explanation about SHAP analysis
│
├── data/                                            # Datasets for training, validation, and testing
│   ├── simulated_data_training_validation.xlsx      # 80/20 split for training and validation
│   ├── matchup_data_test.xlsx                       # Data for model testing
│   ├── brazil_lakes_independent_dataset.xlsx        # Independent test dataset
│   ├── trout_14092025.tif                           # Sentinel-2 mosaic image (first 8 bands) example
│   ├── trout_14092025_cloud.tif                     # Corresponding cloud mask
│   └── README.md                                    # Provides brief explanation about datasets used
│
├── models/                                  # Pre-trained models with hyperparameters and scalers
│   ├── aCDOM440_HPGNN_Model.h5
│   ├── aCDOM440_HPGNN_Model_Hyperparams.json
│   ├── aCDOM440_Scaler.save
│   ├── Chla_HPGNN_Model.h5
│   ├── Chla_HPGNN_Model_Hyperparams.json
│   ├── Chla_Scaler.save
│   ├── Secchi_depth_HPGNN_Model.h5
│   ├── Secchi_depth_HPGNN_Model_Hyperparams.json
│   ├── Secchi_depth_Scaler.save
│   ├── TSS_HPGNN_Model.h5
│   ├── TSS_HPGNN_Model_Hyperparams.json
│   ├── TSS_Scaler.save
│   ├── Turbidity_HPGNN_Model.h5
│   ├── Turbidity_HPGNN_Model_Hyperparams.json
│   ├── Turbidity_Scaler.save
│   └── README.md                            # Provides brief explanation about pre-trained models
│
└── README.md                                # Main README (this file)
```

## About the Work

Objective: To model key optically active water quality parameters in inland lakes using remote sensing data and Hybrid Physics-Guided Neural Network (HPGNN)

Models:

The HPGNN models are developed using feedforward neural network (FFNN) with tunable hyperparameters optimized via the Bayesian Optimization Tuner method, exploring a predefined hyperparameter optimization search space and convergence criteria while training with a custom hybrid loss function. We formulated a custom hybrid loss function combining data fidelity, statistical regularization, and physics-guided constraints to guide the model training and validation for producing stable and physically plausible predictions. Specifically, empirical bio-optical relationships, non-negativity constraints, gradient regularization, smoothness, uncertainty, and variance-based penalties are jointly integrated with data fidelity loss (MSE) to enforce physical plausibility, statistical robustness, and empirical consistency.

The DNN models are also developed to serve as benchmark or baseline models to quantify the added value of hybrid loss function utilized in the HPGNN models and compare performance to highlight the benefits of incorporating physical knowledge into deep learning frameworks for satellite-based water quality retrieval.

Parameters Modeled:

  * Absorption of Colored Dissolved Organic Matter at 440 nm (aCDOM(440))

  * Chlorophyll-a (Chl-a)

  * Secchi Disk Depth (SDD)

  * Total Suspended Solids (TSS)

  * Turbidity

Explainability: SHAP analysis is included to interpret the impact of each spectral band on models predictions.

Data: 

  * GLORIA dataset is used for training, validation and testing of the HPGNN and DNN models
  
  * BRAZA dataset is used for indepnednt testing of the HPGNN models
  
  * Sentinel-2 surface reflectance and imagery are utilized

# Getting Started

## Requirements

Python

tensorflow, keras, keras-tuner, pandas, numpy, scikit-learn, matplotlib, seaborn, rasterio, joblib, shap, matplotlib-scalebar

## Usage

Product Generation

  * Data Preparation: Place your Sentinel-2 images and cloud masks in a folder.
  
  * Generate Products: Use code/product generation/generate_products.py with pre-trained models.

Model Development and Interpretation

  * Data Preparation: Place your training, vlaidation and test datasets in a folder.
  
  * Training Models: Use the scripts in code/dnn models or code/hpgnn models to train models for each parameter.

  * Explainability: Run code/shap analysis/shap_analysis.py to generate SHAP plots for feature importance.


## Citation

Please cite this repository if you use the models or data in your research.
