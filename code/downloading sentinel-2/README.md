# Sentinel-2 Surface Reflectance Extraction and Downloading Image Mosaic & Cloud Mask

## Description

This subdirectory provides scripts to extract and preprocess Sentinel-2 Surface Reflectance (SR) data against water quality in situ measurements (JavaScript Code), and downlaoding Sentinel-2 image mosaic and cloud mask of a lake region (Python Code).

# Python File: Point-based SR Extraction

* Purpose: Extract cloud free SR data for lake points, and compute NDWI to be used for testing of models

* Steps:

    * Load lake coordinates and dates from Excel (Raw data from GLORIA and Brazilian Lakes)

    * Find nearest Sentinel-2 image (±3 days)

    * Extract bands B1–B8, SCL, MSK_CLDPRB

    * Scale reflectance, compute NDWI, check cloud probability

    * Merge with original data and save CSV

* Output: CSV with reflectance, NDWI, and cloud probability.

* Note: It is required to activate and authenticate ee python library

## Requirements

* Python

* ee, pandas, numpy

# JavaScript Code: Lake Region Mosaic & Cloud Mask (Example: Trout Lake, USA)

* Purpose: Download Sentinel-2 image mosaic and cloud mask for a lake region to be used for water quality product generation.

* Steps:

    * Define bounding box and date range

    * Filter Sentinel-2 SR collection and cloud probability

    * Mosaic images and apply cloud mask

    * Visualize RGB and export mosaic & mask as GeoTIFF

* Output: GeoTIFF mosaic and cloud mask

* Note: This code can used on Google Earth Engine Code Editor
