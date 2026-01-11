Models Description: HPGNN models for aCDOM(440) / Chla / Secchi_depth / TSS / Turbidity Estimation

This subdirectory contains codes for Hybrid Physics-Guided Neural Network (HPGNN) models for estimating colored dissolved organic matter absorption at 440 nm (aCDOM(440)), Chlorophyll-a, Secchi Disk Depth, Total Suspended Solids, and Turbidity using Sentinel-2 surface reflectance bands.

The models integrate data fidelity, statistical regularization, and physics-guided constraints to guide the model training and validation for producing stable and physically plausible predictions.

# Input Data

* Input features: Sentinel-2 reflectance bands: B1, B2, B3, B4, B5, B6, B7, B8

* Target variable: aCDOM440 / Chla / Secchi_depth / TSS / Turbidity

* Data source: simulated_data_training_validation and matchup_data_test excel files given in data subdirectory

# Preprocessing

* Outlier removal: Quantile-based filtering (10th–90th percentile) on the target variable

* Normalization: Min–Max scaling of reflectance bands

* Data split: 80% training / 20% validation

# Model Architecture

* Fully connected deep neural network (DNN)

* 2–6 hidden layers (tuned via Bayesian optimization)

* Hidden units: 32–1024 neurons per layer

* Activation: ReLU

* Regularization:

    * Batch Normalization

    * Dropout (0.2–0.5)

* Output: Single neuron predicting target variable

# Physics-Guided Hybrid Loss Function

* The model is trained using a custom hybrid loss function that combines follwoing loss compoents for data-fidelity with physics-guided and statistical regularization constraints:

    1) Mean Squared Error (MSE)
    2) Non-negativity Constraint
    3) Empirical Bio-Optical Constraint
    4) Uncertainty Loss
    5) Variance Loss
    6) Smoothness Loss
    7) Gradient Regularization

* Adaptive Weighting

* Loss weights are dynamically updated during training using a custom callback to gradually strengthen constraints without destabilizing learning.

# Hyperparameter Optimization

* Optimization method: Bayesian Optimization (Keras-Tuner)

* Tuned parameters:

    * Number of layers

    * Neurons per layer

    * Dropout rates

    * Learning rate

    * Loss weights (w₁–w₆)

* Objective: Minimize validation MAE

# Model Training

* Optimizer: Adam

* Batch size: 64

* Epochs: Up to 1000

* Callbacks:

    * Early stopping

    * Learning rate reduction on plateau

    * Adaptive physics-weight adjustment

# Model Evaluation

* Performance is assessed using:

    * R² (Coefficient of Determination)

    * RMSE (Root Mean Squared Error)

    * MAE (Mean Absolute Error)

* Evaluation is conducted on:

    * Training dataset (80%)

    * Validation dataset (20%)

    * Matchup test dataset

# Model Outputs

* Trained HPGNN model (.h5)

* Optimized hyperparameters (.json)

* Performance metrics printed to console

# Requirements

* Python

* tensorflow, keras, keras-tuner, pandas, numpy, scikit-learn

# Key Contribution

This HPGNN framework demonstrates how physics-guided constraints and empirical bio-optical knowledge can be embedded into deep learning models to improve robustness, interpretability, and physical realism in satellite-based water quality retrievals.