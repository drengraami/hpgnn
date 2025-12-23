// bounding box for trout lake, USA
var boundingBox = ee.Geometry.Rectangle(
  [-89.7038, 46.0131, -89.6464, 46.0790], // [Lon_min, Lat_min, Lon_max, Lat_max]
  null,
  false
);

// Use bounding box as region
var region = ee.FeatureCollection(boundingBox);

// Function to get s2cloudless probability mask for one image
function getCloudMask(s2Image) {
  var cloudProb = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filter(ee.Filter.eq('system:index', s2Image.get('system:index')))
    .first();

  var cloudMask = cloudProb.select('probability').lt(10);
  cloudMask = cloudMask.focal_min(3).focal_max(3);
  return cloudMask.rename('cloud_mask');
}

// Load and filter Sentinel-2 SR collection
var s2Col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(region)
  .filterDate("2025-09-04", "2025-09-30")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10));

// --- Mosaic all tiles for the same date ---
var s2Mosaic = s2Col.mosaic().select(['B1','B2','B3','B4','B5','B6','B7','B8']);

var refImage = s2Col.first();

// Get cloud mask based on the reference image ID
var cloudMask = getCloudMask(refImage)
  .reproject({crs: s2Mosaic.projection(), scale: 10}); 

// Visualization
var visParams = {
  bands: ["B4", "B3", "B2"],
  min: 0,
  max: 3000,
  gamma: 1.4
};

Map.centerObject(region, 8);
Map.addLayer(s2Mosaic, visParams, "Sentinel-2 Mosaic");
Map.addLayer(cloudMask, {min:0, max:1, palette:['red','green']}, "Cloud Mask");
Map.addLayer(region, {color: 'FF0000'}, "Region Boundary");

// Export mosaic image
Export.image.toDrive({
  image: s2Mosaic,
  description: "S2_SR_Mosaic",
  scale: 10,
  region: boundingBox,
  fileFormat: "GeoTIFF",
  maxPixels: 1e13
});

// Export cloud mask
Export.image.toDrive({
  image: cloudMask,
  description: "S2_Cloud_Mask_Mosaic",
  scale: 10,
  region: boundingBox,
  fileFormat: "GeoTIFF",
  maxPixels: 1e13
});

// Print metadata
print('Mosaic Date (representative):', refImage.date());
print('Mosaic ID (representative):', refImage.get('system:index'));
