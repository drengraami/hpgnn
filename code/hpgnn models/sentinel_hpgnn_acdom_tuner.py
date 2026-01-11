import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import save_model

# Load Data
file_path = "F:/Paper/Eco_Inf/data/simulated_data_training_validation.xlsx"
data = pd.read_excel(file_path, 'acdom_')  # Load aCDOM440 data

# Reflectance Bands
reflectance_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
features = reflectance_bands
targets = ['aCDOM440']  # Target is aCDOM440

# Remove Outliers using IQR
def remove_outliers(df, features, factor=1.5):
    for feature in features:
        Q1 = df[feature].quantile(0.1)
        Q3 = df[feature].quantile(0.9)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

data = remove_outliers(data, targets)

# Normalize Features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[targets], test_size=0.2, random_state=1
)


# Define Custom Hybrid Loss Function with Adaptive Weights
class AdaptiveWeightsCallback(Callback):
    def __init__(self, w1, w2, w3, w4, w5, w6):
        super().__init__()
        # Initialize weights for all loss components
        self.w1 = w1  # Weight for non-negativity constraint
        self.w2 = w2  # Weight for Empirical bio-optical constraint
        self.w3 = w3  # Weight for uncertainty loss
        self.w4 = w4  # Weight for variance loss
        self.w5 = w5  # Weight for smoothness loss
        self.w6 = w6  # Weight for gradient regularization loss

    def on_epoch_end(self, epoch, logs=None):
        # Adjust weights dynamically
        self.w1 = min(0.1, self.w1 + 0.001)  # Gradually increase non-negativity weight
        self.w2 = min(0.1, self.w2 + 0.001)  # Gradually increase Empirical bio-optical weight
        self.w3 = min(0.1, self.w3 + 0.001)  # Gradually increase uncertainty weight
        self.w4 = min(0.1, self.w4 + 0.001)  # Gradually increase variance weight
        self.w5 = min(0.1, self.w5 + 0.001)  # Gradually increase smoothness weight
        self.w6 = min(0.1, self.w6 + 0.001)  # Gradually increase gradient regularization weight

# Define the custom loss function
def pinn_loss(y_true, y_pred, X_batch, w1, w2, w3, w4, w5, w6):
    # Convert X_batch to a TensorFlow tensor for gradient computation   
    X_train_tf = tf.convert_to_tensor(X_batch, dtype=tf.float32)

    # Standard MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    
    # Empirical bio-optical model loss
    R_b4 = tf.cast(X_batch[:, 3], dtype=tf.float32)  
    R_b3 = tf.cast(X_batch[:, 2], dtype=tf.float32)  
    
    # Bio-optical model coefficients
    a = tf.constant(9.99279839, dtype=tf.float32)
    b = tf.constant(-1.1659923, dtype=tf.float32)    
    
    ratio = tf.math.divide_no_nan(R_b3, R_b4)
    aCDOM440_ciancia = a * tf.exp(b * ratio)

    physics_loss_2 = tf.reduce_mean(tf.square(y_pred - aCDOM440_ciancia))

    # Non-negativity Loss
    physics_loss_non_neg = tf.reduce_mean(tf.square(tf.maximum(-y_pred, 0.0)))
      
    # Gradient regularization loss
    with tf.GradientTape() as tape:
        tape.watch(X_train_tf)
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
    grad = tape.gradient(loss, X_train_tf)
    if grad is not None:
        grad_loss = tf.reduce_mean(tf.abs(grad))
    else:
        grad_loss = 0.0  # Explicitly setting zero loss when no gradient

    # Uncertainity loss
    y_true_uncertainty = tf.reduce_mean(tf.math.reduce_std(y_true, axis=0)) 
    uncertainty_threshold = tf.reduce_mean(y_true) * 0.1  

    if y_true_uncertainty < uncertainty_threshold:
        uncertainty_loss = tf.reduce_mean(tf.math.reduce_std(y_pred, axis=0))  
    else:
        uncertainty_loss = 0.0  # No need to calculate uncertainty loss

    # Variance loss
    y_true_variance = tf.reduce_mean(tf.math.reduce_std(y_true)) 
    variance_threshold = tf.reduce_mean(y_true) * 0.2 

    if y_true_variance < variance_threshold:
        variance_loss = tf.reduce_mean(tf.math.square(y_pred - tf.reduce_mean(y_pred)))  
    else:
        variance_loss = 0.0  # No need to calculate variance loss
    
    # Smoorthness loss
    if tf.rank(y_true) > 1:  
        y_true_smoothness = tf.reduce_mean(tf.abs(y_true[:, 1:] - y_true[:, :-1]))
    else: 
        y_true_smoothness = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

    # Compute smoothness threshold
    smoothness_threshold = y_true_smoothness + tf.math.reduce_std(y_true_smoothness)

    # Compute smoothness loss for predictions only if true values are smooth
    if y_true_smoothness < smoothness_threshold:
        if tf.rank(y_pred) > 1:  
            smoothness_loss = tf.reduce_mean(tf.abs(y_pred[:, 1:] - y_pred[:, :-1]))
        else:  
            smoothness_loss = tf.reduce_mean(tf.abs(y_pred[1:] - y_pred[:-1]))
    else:
        smoothness_loss = 0.0

    # Combine all losses with dynamic weights
    total_loss = (mse_loss + 
                    w1 * physics_loss_non_neg + 
                    w2 * physics_loss_2 + 
                    w3 * uncertainty_loss + 
                    w4 * variance_loss + 
                    w5 * smoothness_loss + 
                    w6 * grad_loss ) 
                    
    
    return total_loss

# Wrapper function to pass the model and weights to the loss function
def custom_loss_wrapper(X_batch, w1, w2, w3, w4, w5, w6):
    def loss(y_true, y_pred):
        return pinn_loss(y_true, y_pred, X_batch, w1, w2, w3, w4, w5, w6)
    return loss

# Define Model with Tunable Hyperparameters
def build_model(hp):
    input_layer = Input(shape=(len(features),))
    x = input_layer

    # Tune number of layers (2 to 6)
    for i in range(hp.Int('num_layers', 2, 6)):
        x = Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=1024, step=32),
            activation='relu'
        )(x)
        x = BatchNormalization()(x)  
        x = Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1))(x)

    # Output layer
    output_layer = Dense(1, name='aCDOM440')(x)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')

    w1 = hp.Float('w1', min_value=0.01, max_value=0.1, step=0.01)  # Non-negativity weight
    w2 = hp.Float('w2', min_value=0.01, max_value=0.1, step=0.01)  # Empirical bio-optical weight
    w3 = hp.Float('w3', min_value=0.01, max_value=0.1, step=0.01)  # Uncertainty weight
    w4 = hp.Float('w4', min_value=0.01, max_value=0.1, step=0.01)  # Variance weight
    w5 = hp.Float('w5', min_value=0.01, max_value=0.1, step=0.01)  # smoothness weight
    w6 = hp.Float('w6', min_value=0.01, max_value=0.1, step=0.01)  # Gradient regularization weight

    # Compile the model with adaptive weights
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=custom_loss_wrapper(X_train.values, w1, w2, w3, w4, w5, w6),
                  metrics=['mae'])

    return model


# Early Stopping to Prevent Overfitting
reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=200, min_lr=1e-6, verbose=1)

def run_tuner(X_train, y_train, X_test, y_test, directory, project_name, seed=1):
    # Define the tuner
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_mae',
        max_trials=25, 
        executions_per_trial=1,
        directory=directory,
        project_name=project_name,
        seed=seed  
    )

    early_stopping = EarlyStopping(monitor='val_mae', patience=200, restore_best_weights=True)

    # Run the tuner
    tuner.search(X_train, y_train,
                 validation_data=(X_test, y_test),
                 epochs=1000,
                 batch_size=64,
                 callbacks=[early_stopping, reduce_lr])

    # Return the tuner object
    return tuner

# Call the function and store the tuner object
tuner = run_tuner(X_train, y_train['aCDOM440'], X_test, y_test['aCDOM440'],
                  directory='F:/Paper/Eco_Inf/tuner', project_name='aCDOM440_sentinel_optimization', seed=42)

# Retrieve Best Hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters Found:")
print(best_hp.values)

# Build Best Model
best_model = tuner.hypermodel.build(best_hp)

# Train Best Model with Adaptive Weights
adaptive_weights_callback = AdaptiveWeightsCallback(
    w1=best_hp.get('w1'), 
    w2=best_hp.get('w2'), 
    w3=best_hp.get('w3'), 
    w4=best_hp.get('w4'), 
    w5=best_hp.get('w5'), 
    w6=best_hp.get('w6')
)

# Summary of the best model
best_model.summary()
early_stopping = EarlyStopping(monitor='val_mae', patience=200, restore_best_weights=True)

# Print dropout rates and weights
dropout_values = [layer.get_config()['rate'] for layer in best_model.layers if isinstance(layer, Dropout)]
print("Dropout Rates:", dropout_values)
print("Non-Negativity Weight (w1):", best_hp.get('w1'))
print("Empirical bio-optical weight (w2):", best_hp.get('w2'))
print("Uncertainty Weight (w3):", best_hp.get('w3'))
print("Variance Weight (w4):", best_hp.get('w4'))
print("smoothness Weight (w5):", best_hp.get('w5'))
print("Gradient Regularization Weight (w6):", best_hp.get('w6'))
print("Learning Rate: ", best_hp.get('learning_rate'))


# Train the best model
history = best_model.fit(X_train, y_train['aCDOM440'], 
                         validation_data=(X_test, y_test['aCDOM440']), 
                         epochs=1000, batch_size=64, verbose=1, 
                         callbacks=[early_stopping, reduce_lr, adaptive_weights_callback])

import json

hyperparams = {
    "learning_rate": best_hp.get('learning_rate'),
    "w1": best_hp.get('w1'), "w2": best_hp.get('w2'), "w3": best_hp.get('w3'), "w4": best_hp.get('w4'),
    "w5": best_hp.get('w5'), "w6": best_hp.get('w6')
}

save_model(best_model, "F:/Paper/Eco_Inf/models/aCDOM440_HPGNN_Model.h5")

with open("F:/Paper/Eco_Inf/models/aCDOM440_HPGNN_Model_Hyperparams.json", "w") as f:
    json.dump(hyperparams, f)

# Load New Data for Testing
new_data_path = "F:/Paper/Eco_Inf/data/matchup_data_test.xlsx"
new_data = pd.read_excel(new_data_path, 'acdom_')

new_data[features] = scaler.transform(new_data[features])
X_new = new_data[features].to_numpy()

# Predict and Evaluate
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
y_pred_test_ = best_model.predict(X_new)

y_pred_train = y_pred_train.flatten()
y_pred_test = y_pred_test.flatten()
y_pred_test_ = y_pred_test_.flatten()

mae_train = mean_absolute_error(y_train['aCDOM440'], y_pred_train)
r2_train = r2_score(y_train['aCDOM440'], y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train['aCDOM440'], y_pred_train))

mae_test = mean_absolute_error(y_test['aCDOM440'], y_pred_test)
r2_test = r2_score(y_test['aCDOM440'], y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test['aCDOM440'], y_pred_test))

mae_test_ = mean_absolute_error(new_data['aCDOM440'], y_pred_test_)
r2_test_ = r2_score(new_data['aCDOM440'], y_pred_test_)
rmse_test_ = np.sqrt(mean_squared_error(new_data['aCDOM440'], y_pred_test_))

print(f"\nTrain R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
print(f"\nTest R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
print(f"\nNew Data R²: {r2_test_:.4f}, RMSE: {rmse_test_:.4f}, MAE: {mae_test_:.4f}")
