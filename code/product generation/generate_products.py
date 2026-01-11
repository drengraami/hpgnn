import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json
from keras.models import load_model
from keras.optimizers import Adam
import tensorflow as tf
import os
import joblib

# Load your custom functions (same as in your original code)
def pinn_loss(y_true, y_pred, model, X_train_tf, w1, w2, w3, w4, w5, w6):
    
    # Standard MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Empirical bio-optical model loss # NDCI for reference
    R_b4 = tf.cast(X_train_tf[:, 3], dtype=tf.float32)  
    R_b5 = tf.cast(X_train_tf[:, 4], dtype=tf.float32)  
    
    R_b5_b4=  tf.cast((R_b5 - R_b4), dtype=tf.float32)
    R_b5b4=  tf.cast((R_b5 + R_b4), dtype=tf.float32)
    
    # Bio-optical model coefficients
    d = tf.constant(28.79692743, dtype=tf.float32)
    e = tf.constant(147.9780212, dtype=tf.float32)
    f = tf.constant(135.97716937, dtype=tf.float32)

    ndci = tf.math.divide_no_nan(R_b5_b4, R_b5b4)  # Avoid NaN issues

    chl_a_ndci = d + e * ndci + f * tf.square(ndci)

    physics_loss_2 = tf.reduce_mean(tf.square(y_pred - chl_a_ndci))
    
    # Non-negativity loss
    physics_loss_non_neg = tf.reduce_mean(tf.square(tf.maximum(-y_pred, 0.0)))
          
    # Gradient regularization loss
    with tf.GradientTape() as tape:
        tape.watch(X_train_tf)
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
    grad = tape.gradient(loss, X_train_tf)
    if grad is not None:
        grad_loss = tf.reduce_mean(tf.abs(grad))
    else:
        grad_loss = 0.0

    # uncertainity loss
    y_true_uncertainty = tf.reduce_mean(tf.math.reduce_std(y_true, axis=0))
    uncertainty_threshold = tf.reduce_mean(y_true) * 0.1

    if y_true_uncertainty < uncertainty_threshold:
        uncertainty_loss = tf.reduce_mean(tf.math.reduce_std(y_pred, axis=0))
    else:
        uncertainty_loss = 0.0

    # variance loss
    y_true_variance = tf.reduce_mean(tf.math.reduce_std(y_true))
    variance_threshold = tf.reduce_mean(y_true) * 0.2

    if y_true_variance < variance_threshold:
        variance_loss = tf.reduce_mean(tf.math.square(y_pred - tf.reduce_mean(y_pred)))
    else:
        variance_loss = 0.0
    
    # smoothness loss
    if tf.rank(y_true) > 1:
        y_true_smoothness = tf.reduce_mean(tf.abs(y_true[:, 1:] - y_true[:, :-1]))
    else:
        y_true_smoothness = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

    smoothness_threshold = y_true_smoothness + tf.math.reduce_std(y_true_smoothness)

    if y_true_smoothness < smoothness_threshold:
        if tf.rank(y_pred) > 1:
            smoothness_loss = tf.reduce_mean(tf.abs(y_pred[:, 1:] - y_pred[:, :-1]))
        else:
            smoothness_loss = tf.reduce_mean(tf.abs(y_pred[1:] - y_pred[:-1]))
    else:
        smoothness_loss = 0.0
    
    total_loss = (mse_loss + 
                w1 * physics_loss_non_neg + 
                w2 * physics_loss_2 + 
                w3 * uncertainty_loss + 
                w4 * variance_loss + 
                w5 * smoothness_loss + 
                w6 * grad_loss)
    return total_loss

def custom_loss_wrapper(model, X_train_tf, w1, w2, w3, w4, w5, w6):
    def loss(y_true, y_pred):
        return pinn_loss(y_true, y_pred, model, X_train_tf, w1, w2, w3, w4, w5, w6)
    return loss

# 1. Load and preprocess the Sentinel-2 image
def load_and_preprocess_image(image_path, cloud_mask_path=None):
    """Load Sentinel-2 image, cloud mask, and prepare features"""
    with rasterio.open(image_path) as src:
        bands = src.read()
        metadata = src.meta
        height, width = bands.shape[1], bands.shape[2]
        
        
        band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
        band_dict = {name: bands[i] for i, name in enumerate(band_names)}
    
    # Load cloud mask if provided
    cloud_mask = None
    if cloud_mask_path:
        with rasterio.open(cloud_mask_path) as src:
            cloud_mask = src.read(1)
            # Ensure cloud mask matches image dimensions
            if cloud_mask.shape != (height, width):
                cloud_mask = None
                print("Warning: Cloud mask dimensions don't match image")
    
    # Scale reflectance values (divide by 10000)
    scaled_bands = {name: band/31420.0 for name, band in band_dict.items()}
    
    # Compute NDWI (Normalized Difference Water Index)
    ndwi = (scaled_bands['B3'] - scaled_bands['B8']) / (scaled_bands['B3'] + scaled_bands['B8'])

    # Water mask: NDWI > 0 (adjust threshold if needed)
    water_mask = ndwi > 0

    # Prepare features as pandas DataFrame with column names
    X = pd.DataFrame({
        'B1': scaled_bands['B1'].flatten(),
        'B2': scaled_bands['B2'].flatten(),
        'B3': scaled_bands['B3'].flatten(),
        'B4': scaled_bands['B4'].flatten(),
        'B5': scaled_bands['B5'].flatten(),
        'B6': scaled_bands['B6'].flatten(),
        'B7': scaled_bands['B7'].flatten(),
        'B8': scaled_bands['B8'].flatten()
    })

    # Apply water mask & filter NaN values
    nan_mask = X.isnull().any(axis=1)  # Identify NaN rows
    valid_mask = water_mask.flatten() & ~nan_mask  # Keep only water pixels
    
    # Apply cloud mask if available
    if cloud_mask is not None:
        # Cloud mask should be 0=cloud, 1=clear
        cloud_mask_flat = cloud_mask.flatten() == 1
        valid_mask = valid_mask & cloud_mask_flat
    
    return X, metadata, height, width, valid_mask

# 2. Load your trained models and scalers
def load_models_and_scalers():
    
    target_mapping = {
        'chla': {
            'hyperparams': 'F:/Paper/Eco_Inf/models/Chla_HPGNN_Model_Hyperparams.json',
            'model_path': 'F:/Paper/Eco_Inf/models/Chla_HPGNN_Model.h5',   
            'scaler_path': 'F:/Paper/Eco_Inf/models/Chla_Scaler.save'
         
        },
        'tss': {
            'hyperparams': 'F:/Paper/Eco_Inf/models/TSS_HPGNN_Model_Hyperparams.json',
            'model_path': 'F:/Paper/Eco_Inf/models/TSS_HPGNN_Model.h5',
            'scaler_path': 'F:/Paper/Eco_Inf/models/TSS_Scaler.save'
        },
        'acdom': {
            'hyperparams': 'F:/Paper/Eco_Inf/models/aCDOM440_HPGNN_Model_Hyperparams.json',
            'model_path': 'F:/Paper/Eco_Inf/models/aCDOM440_HPGNN_Model.h5',
            'scaler_path': 'F:/Paper/Eco_Inf/models/aCDOM440_Scaler.save'
        }, 
        'secchi_depth': {        
            'hyperparams': 'F:/Paper/Eco_Inf/models/Secchi_depth_HPGNN_Model_Hyperparams.json',
            'model_path': 'F:/Paper/Eco_Inf/models/Secchi_depth_HPGNN_Model.h5',  
            'scaler_path': 'F:/Paper/Eco_Inf/models/Secchi_depth_Scaler.save'
        },
        'turbidity': {
            'hyperparams': 'F:/Paper/Eco_Inf/models/Turbidity_HPGNN_Model_Hyperparams.json',
            'model_path': 'F:/Paper/Eco_Inf/models/Turbidity_HPGNN_Model.h5', 
            'scaler_path': 'F:/Paper/Eco_Inf/models/Turbidity_Scaler.save'
        }
    }
    models = {}
    scalers = {}
    
    for target, config in target_mapping.items():
        # Load hyperparameters
        with open(config['hyperparams'], 'r') as f:
            hyperparams = json.load(f)
        
        # Load model
        model = load_model(config['model_path'], compile=False)
        
        # Compile with custom loss
        optimizer = Adam(learning_rate=hyperparams['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss=custom_loss_wrapper(
                model, None,
                hyperparams['w1'], hyperparams['w2'], hyperparams['w3'],
                hyperparams['w4'], hyperparams['w5'], hyperparams['w6']
            )
        )
        
        # Load scaler (you'll need to save these during training)
        # For now we'll create a new one - replace with your actual saved scalers
        scaler = MinMaxScaler()
        scaler = joblib.load(config['scaler_path'])
        models[target] = model
        scalers[target] = scaler
    
    return models, scalers

# 3. Main processing function
def process_image(image_path, output_dir, cloud_mask_path=None):
    """Process Sentinel-2 image with all models"""
    # Load and preprocess image with cloud mask
    X_df, metadata, height, width, valid_mask = load_and_preprocess_image(
        image_path, 
        cloud_mask_path
    )
    
    # Load models and scalers
    models, scalers = load_models_and_scalers()
    
    # Initialize output arrays
    results = {
        'chla': np.full((height * width,), np.nan),
        'tss': np.full((height * width,), np.nan),
        'acdom': np.full((height * width,), np.nan),
        'secchi_depth': np.full((height * width,), np.nan),
        'turbidity': np.full((height * width,), np.nan)
    }

    # Select only water pixels for processing
    X_valid = X_df[valid_mask]

    
    # Process with each model
    for target, model in models.items():
        scaler = scalers[target]

        # Normalize input features
        X_scaled = scaler.transform(X_valid)

        # Predict
        y_pred = model.predict(X_scaled).flatten()

        # Store predictions only in valid water pixel positions
        results[target][valid_mask] = y_pred

    # Reshape results into image dimensions
    for target in results:
        results[target] = results[target].reshape((height, width))
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    for target, data in results.items():
        output_path = os.path.join(output_dir, f'{target}_prediction.tif')
        save_prediction(data, metadata, output_path)
    
    # Create visualization
    visualize_results(results, image_path, os.path.join(output_dir, 'water_quality_predictions.png'))
    
    return results

def save_prediction(data, metadata, output_path):
    """Save prediction as GeoTIFF"""
    metadata.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': np.nan
    })
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(data.astype(np.float32), 1)

import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def visualize_results(results, image_path, output_path):
    """Create visualization of all predictions with RGB background"""
    # Load the original image with geospatial info
    with rasterio.open(image_path) as src:
        rgb_bands = src.read([4, 3, 2])  # B4, B3, B2 for RGB
        rgb_bands = np.clip(rgb_bands / 10000.0, 0, 1)  # Scale and clip reflectance values
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        crs = src.crs.to_string() if src.crs else 'EPSG:4326'

    # Create figure with 5 columns (for all parameters)
    fig, axes = plt.subplots(1, 5, figsize=(40, 8))
    
    # Function to create transparent overlay
    def create_overlay(base, overlay_data, cmap, vmax):
        norm = plt.Normalize(vmin=0, vmax=vmax)
        cmap = plt.colormaps.get_cmap(cmap)
        colored_data = cmap(norm(overlay_data))
        valid_mask = ~np.isnan(overlay_data)
        alpha = np.where(valid_mask, 0.7, 0)
        colored_data[..., -1] = alpha
        result = base.copy()
        result[valid_mask] = (1 - alpha[valid_mask])[..., np.newaxis] * base[valid_mask] + alpha[valid_mask][..., np.newaxis] * colored_data[valid_mask, :3]
        return result
    
    # Create RGB base image
    rgb_stretched = np.clip((rgb_bands * 3.5), 0, 1).transpose(1, 2, 0)
    
    # Parameters to plot
    params = [
        ('chla', 'viridis', np.nanpercentile(results['chla'], 95), 'Chlorophyll-a (mg/m³)'),
        ('tss', 'plasma', np.nanpercentile(results['tss'], 95), 'Total Suspended Solids (g/m³)'),
        ('acdom', 'cividis', np.nanpercentile(results['acdom'], 95), 'aCDOM(440) (m⁻¹)'),
        ('secchi_depth', 'magma', np.nanpercentile(results['secchi_depth'], 95), 'Secchi Depth (m)'),
        ('turbidity', 'inferno', np.nanpercentile(results['turbidity'], 95), 'Turbidity (NTU)')
    ]
    
    for ax, (param, cmap, vmax, title) in zip(axes, params):
        # Plot RGB background
        ax.imshow(rgb_stretched, extent=extent)
        
        # Overlay prediction data
        overlay = create_overlay(rgb_stretched, results[param], cmap, vmax)
        ax.imshow(overlay, extent=extent)
        
        # Add map elements
        ax.set_title(title, fontsize=12)
        
        # Add scale bar
        scale_length = calculate_scale_length(extent)
        ax.add_artist(ScaleBar(scale_length, location='lower left', box_alpha=0.5))
        
        # Add north arrow
        ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.88),
                   arrowprops=dict(facecolor='black', width=2, headwidth=8),
                   ha='center', va='center', fontsize=12,
                   xycoords='axes fraction')
        
        # Removed grid lines (this line was deleted)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title.split(' (')[0], rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', transparent=True)
    plt.close()

def calculate_scale_length(extent):
    """Calculate appropriate scale bar length based on image extent"""
    width_deg = extent[1] - extent[0]
    # Approximate conversion from degrees to meters at equator
    width_m = width_deg * 111320
    if width_m < 5000:
        return 1000  # 1 km
    elif width_m < 20000:
        return 5000  # 5 km
    else:
        return 10000  # 10 km

# Run the processing
if __name__ == "__main__":
    image_path = "F:/Paper/Eco_Inf/data/trout_14092025.tif"  # Your downloaded image
    cloud_mask_path = "F:/Paper/Eco_Inf/data/trout_14092025_cloud.tif"
    output_dir = "F:/Paper/Eco_Inf/product/trout_14092025"
    
    print("Starting image processing...")
    results = process_image(image_path, output_dir)
   
   