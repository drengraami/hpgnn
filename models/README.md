# Pre-Trained HPGNN Models

## Description

This subdirectory contains the pre-trained models, hyperparameters, and scalers for water quality parameters. These models were developed using the provided training datasets and can be used to generate water quality products.

## Contents

* For each water quality parameter (aCDOM(440), Chla, Secchi_depth, TSS, Turbidity), the following files are provided:

* File	Description
    *_HPGNN_Model.h5	Trained HPGNN model (Keras .h5 format)
    *_HPGNN_Model_Hyperparams.json	JSON file containing the hyperparameters and loss function weights used for training
    *_Scaler.save	Saved MinMaxScaler object for normalizing input features

## Usage

* These models can be used to generate water quality products using the python script given in the product generation subdirectory. 

## Notes

* Ensure that the input data is preprocessed and scaled consistently with the training data.

* The predictions are stored as GeoTIFF images and visualizations are created automatically using the provided scripts.
