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

# Define Reflectance Bands
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
        x = BatchNormalization()(x)  # Added BatchNormalization
        x = Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1))(x)

    # Output layer
    output_layer = Dense(1, name='aCDOM440')(x)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')

    # Compile the model with adaptive weights
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])

    return model


# Early Stopping to Prevent Overfitting
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, min_lr=1e-6, verbose=1)

def run_tuner(X_train, y_train, X_test, y_test, directory, project_name, seed=1):
    # Define the tuner
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=10,  # Number of trials
        executions_per_trial=1,
        directory=directory,
        project_name=project_name,
        seed=seed  # Optional: Set a random seed for reproducibility
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

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
                  directory='F:/Paper/Eco_Inf/tuner/dnn', project_name='aCDOM440_sentinel_optimization', seed=42)

# Retrieve Best Hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters Found:")
print(best_hp.values)

# Build Best Model
best_model = tuner.hypermodel.build(best_hp)

# Summary of the best model
best_model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

# Print dropout rates and weights
dropout_values = [layer.get_config()['rate'] for layer in best_model.layers if isinstance(layer, Dropout)]
print("Dropout Rates:", dropout_values)
print("Learning Rate: ", best_hp.get('learning_rate'))

# Train the best model
history = best_model.fit(X_train, y_train['aCDOM440'], 
                         validation_data=(X_test, y_test['aCDOM440']), 
                         epochs=1000, batch_size=64, verbose=1, 
                         callbacks=[early_stopping, reduce_lr])

save_model(best_model, "F:/Paper/Eco_Inf/models/aCDOM440_DNN_Model.h5")

# Load New Data for Testing
new_data_path = "F:/Paper/Eco_Inf/data/matchup_data_test.xlsx"
new_data = pd.read_excel(new_data_path, 'acdom_')

new_data[features] = scaler.transform(new_data[features])
X_new = new_data[features].to_numpy()

# Predict and Evaluate
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
y_pred_test_ = best_model.predict(X_new)

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




