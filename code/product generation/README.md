# Sentinel-2 Water Quality Predictions with the HPGNN Models

## Description

This subdirectory contains script to predict multiple water quality parameters (aCDOM(440), Chla, Secchi_depth, TSS, and Turbidity) from Sentinel-2 imagery and generate products using pretrained HPGNN models.

## Purpose

* Predict aCDOM(440), Chla, Secchi_depth, TSS, and Turbidity for water bodies.

* Handle cloud masking and water pixel selection.

* Generate GeoTIFF outputs and RGB overlay visualizations.

## Workflow

1) Load and preprocess Sentinel-2 images (Given in data subdirectory) 

    * Scale reflectance (B1–B8)

    * Compute NDWI to identify water pixels

    * Apply optional cloud mask

2) Load pretrained models and scalers (Given in models subdirectory)

    * The HPGNN models for all target parameters

    * Load saved hyperparameters and MinMaxScalers

3) Predict water quality

    * Normalize water pixels

    * Generate predictions for valid water pixels

    * Reshape predictions into original image dimensions

4) Save outputs

    * GeoTIFF files for each parameter: <parameter>_prediction.tif

    * RGB overlay visualization with color-mapped predictions

## Requirements

* Python

* tensorflow, keras, numpy, pandas, rasterio, matplotlib, scikit-learn, joblib, matplotlib_scalebar

## Output

* GeoTIFF files for each predicted water quality parameter

* Visualization PNG with all parameters overlaid on RGB base
