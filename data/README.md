# Introduction
Data Subdirectory for Water Quality Inversion

This subdirectory contains all the datasets used for training, testing, and demonstration of water quality prediction from Sentinel-2 imagery.

# Excel Files

* Each Excel file contains six sheets:

* acdom_ – Data for absorption of Colored Dissolved Organic Matter at 440 nm

* chla_ – Data for Chlorophyll-a concentration

* sdd_ – Data for Secchi Disk Depth

* tss_ – Data for Total Suspended Solids

* turb_ – Data for Turbidity

* meta_data – Column definitions and description of data available in the sheets

# Escel Files Data Purpose

* simulated_data_training_validation.xlsx: Used for training and validation of HPGNN and DNN models (80% training, 20% validation)

    * It contains simulated Sentinel-2 remote sensing reflectance and corresponding GLORIA in situ information trasnformed from GLORIA hyperspectral remote sensing reflectance using Sentinel-2 spectral response functions

* matchup_data_test.xlsx: Used as test dataset to evaluate models' performance on matchup Sentinel-2 surface reflectnace 
    
    * It contains matchup Sentinel-2 surface reflectance against corresponding GLORIA in situ information extracted using python code given in downloading sentinel-2 subdirectory

* brazil_lakes_independent_dataset.xlsx: Used as independent test dataset for additional validation and benchmarking

    * It contains matchup Sentinel-2 surface reflectance against BRAZA - a bio-optical database for the remote sensing of water quality in BRAZil coAstal and inland waters extracted using python code given in downloading sentinel-2 subdirectory

* Each sheet contains spectral reflectance bands (B1–B8) and target water quality parameters, along with other metadata columns as described in meta_data.

# TIFF Files (Example: Trout Lake, USA)

* trout_14092025.tif: Mosaic image of Sentinel-2 bands 1–8 used for demonstration of water quality mapping

* trout_14092025_cloud.tif: Cloud probability mask for the mosaic (0 = cloud, 1 = clear)

* These files are provided as example inputs for the water quality prediction generation using python code given in generate_products.

# Notes

* The Excel files are ready for direct use with the HPGNN and DNN models.

* TIFF files are example Sentinel-2 images for demonstration purposes; processing with cloud masking is supported in the pipeline.

* All datasets are structured to maintain consistent column naming across sheets and files.