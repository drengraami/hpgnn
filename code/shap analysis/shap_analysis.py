import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
import shap
import json
from keras.optimizers import Adam
import tensorflow as tf

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

# Wrapper function to pass the model and weights to the loss function
def custom_loss_wrapper(model, X_train_tf, w1, w2, w3, w4, w5, w6):
    def loss(y_true, y_pred):
        return pinn_loss(y_true, y_pred, model, X_train_tf, w1, w2, w3, w4, w5, w6)
    return loss

# Function to remove outliers using IQR method
def remove_outliers(df, features, factor=1.5):
    removed_count = {}
    for feature in features:
        Q1 = df[feature].quantile(0.1)
        Q3 = df[feature].quantile(0.9)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Count how many values are removed before filtering
        before_count = len(df)
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
        after_count = len(df)
        
        # Track the number of removed values
        removed_count[feature] = before_count - after_count
        
    return df, removed_count

# Define file path and features
file_path = "F:/Paper/Eco_Inf/data/simulated_data_training_validation.xlsx"
file_path_new = "F:/Paper/Eco_Inf/data/matchup_data_test.xlsxx"
reflectance_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
features = reflectance_bands 

target_mapping = {
    'acdom': {
        'sheet_name': 'acdom_',
        'column_name': 'aCDOM440',
        'hyperparams': 'F:/Paper/Eco_Inf/models/aCDOM440_HPGNN_Model_Hyperparams.json',
        'project_name': 'F:/Paper/Eco_Inf/models/aCDOM440_HPGNN_Model.h5',
        'title': r'$\mathbf{a_{\mathbf{CDOM}}(440)}$',
        'color': 'green'
    },
     'chla': {
        'sheet_name': 'chla_',
        'column_name': 'Chla',
        'hyperparams': 'F:/Paper/Eco_Inf/models/Chla_HPGNN_Model_Hyperparams.json',
        'project_name': 'F:/Paper/Eco_Inf/models/Chla_HPGNN_Model.h5', 
        'title': 'Chl-a',
        'color': 'darkorange'
    },
    'Secchi_depth': {
        'sheet_name': 'sdd_', 
        'column_name': 'Secchi_depth',  
        'hyperparams': 'F:/Paper/Eco_Inf/models/Secchi_depth_HPGNN_Model_Hyperparams.json',
        'project_name': 'F:/Paper/Eco_Inf/models/Secchi_depth_HPGNN_Model.h5',  
        'title': 'SDD',
        'color': 'royalblue'
    },
    'tss': {
        'sheet_name': 'tss_',
        'column_name': 'TSS',
        'hyperparams': 'F:/Paper/Eco_Inf/models/TSS_HPGNN_Model_Hyperparams.json',
        'project_name': 'F:/Paper/Eco_Inf/models/TSS_HPGNN_Model.h5', 
        'title': 'TSS',
        'color': 'purple'
    },
    'turbidity': {
        'sheet_name': 'turb_',
        'column_name': 'Turbidity',
        'hyperparams': 'F:/Paper/Eco_Inf/models/Turbidity_HPGNN_Model_Hyperparams.json',
        'project_name': 'F:/Paper/Eco_Inf/models/Turbidity_HPGNN_Model.h5', 
        'title': 'Turbidity',
        'color': 'crimson'
    }
}

# Load data and preprocess
data = {}
results = {}
shap_values_dict = {}


for i, (target, mapping) in enumerate(target_mapping.items()):
    # Load data
    df = pd.read_excel(file_path, mapping['sheet_name'])
    
    # Remove outliers
    df, _ = remove_outliers(df, [mapping['column_name']])
    
    # Normalize/scale features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[mapping['column_name']], test_size=0.2, random_state=1)
    
    # Load the best model
    best_model = load_model(mapping['project_name'], compile=False)
    with open(mapping['hyperparams'], "r") as f:
        hyperparams = json.load(f)

    learning_rate = hyperparams["learning_rate"]
    w1, w2, w3, w4, w5, w6 =(
        hyperparams["w1"], hyperparams["w2"], hyperparams["w3"],
        hyperparams["w4"], hyperparams["w5"], hyperparams["w6"]        
    )

    optimizer = Adam(learning_rate=learning_rate)
    best_model.compile(optimizer=optimizer, loss=pinn_loss)
    
    # Convert X_train to a NumPy array for SHAP
    X_train_array = X_train.values
    X_test_array = X_test.values

    sample_size = min(1000, X_train_array.shape[0])
    background = X_train_array[np.random.RandomState(42).choice(
        X_train_array.shape[0], 
        sample_size, 
        replace=False
    )]

    # Initialize SHAP DeepExplainer
    explainer = shap.DeepExplainer(best_model, background)

    # Compute SHAP values for the test set
    shap_values = explainer.shap_values(X_test_array)
        
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  

    if len(shap_values.shape) == 3:
        shap_values = np.squeeze(shap_values, axis=-1)  

    # Plot SHAP summary plot for feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_array, feature_names=features, show=False)
    plt.xlabel('SHAP Value (Unitless)', fontsize=18)
    plt.ylabel('Spectral Bands', fontsize=18)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14) 

    fig = plt.gcf()
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel(
            "Scaled Spectral Band Reflectance",
            fontsize=14,
            
    )
    plt.tight_layout()
    
    # Save the SHAP summary plot
    output_path = f"F:/Paper/Eco_Inf/shap_plots/global_shap_bar_{target}_HPGNN.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')

    # Calculate mean absolute SHAP values for each feature
    mean_shap = np.abs(shap_values).mean(axis=0)  

    shap_df = pd.DataFrame({
        'Feature': features,
        'Mean_SHAP': mean_shap
    }).sort_values('Mean_SHAP', ascending=False)

    # Plot mean SHAP values
    plt.figure(figsize=(10, 6))
    sns.barplot(data=shap_df, x='Mean_SHAP', y='Feature', palette='viridis')    
    plt.xlabel('MMean |SHAP value| (Unitless)', fontsize=18)
    plt.ylabel('Spectral Bands', fontsize=18) 

    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)  
    plt.tight_layout()

    # Save the plot
    mean_shap_path = f"F:/Paper/Eco_Inf/shap_plots/global_shap_beeswarm_{target}_HPGNN.png"
    plt.savefig(mean_shap_path, dpi=600, bbox_inches='tight')
    plt.close()

    shap_values_dict[target] = shap_values
    y_test_array = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    # Select representative percentiles
    percentiles = {
        "Low (P10)": np.percentile(y_test_array, 10),
        "Median (P50)": np.percentile(y_test_array, 50),
        "High (P90)": np.percentile(y_test_array, 90),
    }

    selected_indices = {}
    for label, value in percentiles.items():
        idx = np.argmin(np.abs(y_test_array - value))
        selected_indices[label] = idx

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (label, idx) in zip(axes, selected_indices.items()):

        sample_shap = shap_values[idx]

        shap_df_local = pd.DataFrame({
            "Feature": features,
            "SHAP Value": sample_shap
        }).sort_values("SHAP Value")

        colors = shap_df_local["SHAP Value"].apply(
            lambda v: "crimson" if v > 0 else "royalblue"
        )

        ax.barh(
            shap_df_local["Feature"],
            shap_df_local["SHAP Value"],
            color=colors
        )

        ax.axvline(0, color="black", linewidth=0.8)

        vmax = np.max(np.abs(sample_shap))
        ax.set_xlim(-1.1 * vmax, 1.1 * vmax)

        ax.set_title(
            f"{label} Concentration Sample",
            fontsize=13,
            fontweight="bold"
        )
        ax.set_xlabel("SHAP Contribution (Unitless)")

    axes[0].set_ylabel("Spectral Bands")
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = (
        f"F:/Paper/Eco_Inf/shap_plots/"
        f"local_shap_bar_{target}_HPGNN.png"
    )

    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


    print(f"Completed SHAP analysis for {target}.")

    print(f"\n{'='*60}")
    print(f"SHAP ANALYSIS SUMMARY: {mapping['title']} ({target})")
    print(f"{'='*60}")

    # 1. Calculate and print overall summary statistics
    mean_abs_shap = np.mean(np.abs(shap_values))
    std_abs_shap = np.std(np.abs(shap_values))
    print(f"Mean |SHAP value|: {mean_abs_shap:.4f}")
    print(f"Std |SHAP value|: {std_abs_shap:.4f}")
    print(f"Sample size (test set): {X_test_array.shape[0]}")

    mean_abs_shap_by_feature = np.abs(shap_values).mean(axis=0)

    sorted_indices = np.argsort(mean_abs_shap_by_feature)[::-1]

    print(f"\nFeature Importance Ranking (by mean |SHAP|):")
    print("-" * 50)
    for rank, idx in enumerate(sorted_indices, 1):
        feature_name = features[idx]
        importance_value = mean_abs_shap_by_feature[idx]
        print(f"  {rank:2d}. {feature_name:>3}: {importance_value:.6f}")

    mean_raw_shap_by_feature = shap_values.mean(axis=0)
    print(f"\nDirectionality of Effect for Top Features:")
    print("-" * 50)

    top_n = min(8, len(features))
    for idx in sorted_indices[:top_n]:
        feature_name = features[idx]
        mean_abs_val = mean_abs_shap_by_feature[idx]
        mean_raw_val = mean_raw_shap_by_feature[idx]
        direction = "Positive" if mean_raw_val > 0 else "Negative"
        print(f"  {feature_name:>3}: Mean SHAP = {mean_raw_val:+.6f} ({direction} impact on {mapping['title']})")

    print(f"{'='*60}\n")
    print(f"Completed SHAP analysis for {target}.")
