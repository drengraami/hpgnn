import ee
import pandas as pd

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-surfacereflectance')

# Load the Excel file
excel_path = "F:/Paper/Eco_Inf/data/GLORIA_Lakes_data.xlsx"
df = pd.read_excel(excel_path, "GLORIA_Lakes_data")

# Convert DataFrame to Earth Engine FeatureCollection
def dataframe_to_ee_featurecollection(df):
    features = []
    for index, row in df.iterrows():
        if pd.isna(row['Longitude']) or pd.isna(row['Latitude']):
            print(f"Skipping row {index} due to missing coordinates.")
            continue
        point = ee.Geometry.Point(row['Longitude'], row['Latitude'])
        feature = ee.Feature(point, row.to_dict())
        features.append(feature)
    return ee.FeatureCollection(features)

ee_fc = dataframe_to_ee_featurecollection(df)

# Define the Sentinel-2 Surface Reflectance dataset (COPERNICUS/S2_SR_HARMONIZED)
sentinel2_sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

# Function to find the nearest Sentinel-2 SR image within ±3 days
def find_nearest_image(point, target_date):
    target_date_ee = ee.Date(target_date)
    start_date = target_date_ee.advance(-3, 'day')
    end_date = target_date_ee.advance(3, 'day')

    filtered_collection = sentinel2_sr.filterDate(start_date, end_date).filterBounds(point)
    count = filtered_collection.size().getInfo()
    if count == 0:
        print(f"No images found for point {point} from {start_date.format('YYYY-MM-dd').getInfo()} to {end_date.format('YYYY-MM-dd').getInfo()}")
        return None

    def add_date_diff(image):
        date_diff = ee.Number(image.date().difference(target_date_ee, 'day')).abs()
        return image.set('date_diff', date_diff)

    sorted_collection = filtered_collection.map(add_date_diff).sort('date_diff')
    nearest_image = sorted_collection.first()
    
    if nearest_image.getInfo() is None:
        print(f"No valid image found for point {point} on {target_date}")
        return None
    
    return nearest_image


# Function to compute NDWI
def compute_ndwi(reflectance):
    green = reflectance.get('B3')
    nir = reflectance.get('B8')

    if green is None or nir is None or (green + nir) == 0:
        return None
    return (green - nir) / (green + nir)

# Function to extract Sentinel-2 SR data for a point, including SCL filtering
def extract_sentinel2_sr_data(point, date_str, row, ndwi_threshold=0.2):
    target_date = date_str.split('T')[0]  
    nearest_image = find_nearest_image(point, target_date)
    if nearest_image is None:
        return None
    
    try:
        
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'SCL', 'MSK_CLDPRB']  
        extract = nearest_image.select(bands).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10,  # Sentinel-2 resolution is 10m for most bands
            maxPixels=1e9
        ).getInfo()

        if not extract:
            print(f"No valid reflectance data for point {point} on {date_str}")
            return None

        # Apply scaling factor to SR bands: Multiply by 0.0001
        for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']:
            if band in extract:
                extract[band] = extract[band] * 0.0001


        # Compute NDWI
        ndwi = compute_ndwi(extract)
        
        # Check cloud probability
        cloud_prob = extract.get('MSK_CLDPRB', 100)  # Default to 100 if not available

        # Add computed NDWI and date information
        extract['NDWI'] = ndwi
        extract['cloud_probability'] = cloud_prob
        extract['image_date'] = nearest_image.date().format('YYYY-MM-dd').getInfo()
        extract['target_date'] = target_date

        # Merge with original row data
        for col in row.index:
            extract[col] = row[col]

        print(f"Extracted clear water pixel at {point} on {target_date}: {extract}")
        return extract
    except Exception as e:
        print(f"Error processing Sentinel-2 SR image for point {point} on {date_str}: {e}")
        return None


results = []

try:

    for index, row in df.iterrows():
        if pd.isna(row['Longitude']) or pd.isna(row['Latitude']):
            continue
        point = ee.Geometry.Point(row['Longitude'], row['Latitude'])
        date_str = row['Date_Time_UTC']
        reflectance_data = extract_sentinel2_sr_data(point, date_str, row, ndwi_threshold=0.2)
        if reflectance_data:
            results.append(reflectance_data)

    # Convert results to DataFrame and save
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("F:/Paper/Eco_Inf/data/sr_intermediate.csv", index=False)
        print("Sentinel-2 water pixel Intermediate data saved successfully.")
    else:
        print("No valid water pixels extracted.")

except Exception as e:
    print(f"An error occurred during processing: {e}")
finally:
    # Convert results to DataFrame and save
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("F:/Paper/Eco_Inf/data/sr.csv", index=False)
        print("Sentinel-2 water pixel data saved successfully.")
    else:
        print("No valid water pixels extracted.")




    