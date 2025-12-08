# A2 - Time Series Forecasting with LSTM
# International Airline Passengers Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("Loading dataset...")

# Load the International Airline Passengers dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)

print("First 5 rows of the dataset:")
print(data.head())

# Use only the 'Passengers' column
passengers = data["Passengers"].values.astype(float).reshape(-1, 1)

# Normalize data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
passengers_scaled = scaler.fit_transform(passengers)

# Function to create sequences for LSTM
def create_dataset(dataset, time_step=12):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step, 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 12  # use last 12 months to predict next month

# Train-test split (80% train, 20% test)
train_size = int(len(passengers_scaled) * 0.8)
train_data = passengers_scaled[:train_size]
test_data = passengers_scaled[train_size - time_step:]  # overlap for sequence

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to [samples, time_steps, features] for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("Building LSTM model...")

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

print("Making predictions...")
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and true values
train_predict_inv = scaler.inverse_transform(train_predict)
test_predict_inv = scaler.inverse_transform(test_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Prepare full series for plotting
full_series = passengers

# Create arrays for plotting train and test predictions on the full timeline
train_plot = np.empty_like(full_series, dtype="float32")
train_plot[:] = np.nan
train_plot[time_step:time_step + len(train_predict_inv)] = train_predict_inv.reshape(-1)

test_plot = np.empty_like(full_series, dtype="float32")
test_plot[:] = np.nan
test_plot[train_size:train_size + len(test_predict_inv)] = test_predict_inv.reshape(-1)

print("Plotting results...")

plt.figure(figsize=(12, 6))
plt.plot(full_series, label="Actual Passengers")
plt.plot(train_plot, label="Train Prediction")
plt.plot(test_plot, label="Test Prediction")
plt.title("International Airline Passengers Forecasting using LSTM")
plt.xlabel("Time (Months)")
plt.ylabel("Number of Passengers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lstm_forecast.png")
plt.show()

# Simple evaluation: print last few true vs predicted values
print("\nLast 10 Actual vs Predicted (Test set):")
for i in range(-10, 0):
    print(f"Actual: {int(y_test_inv[i, 0])}, Predicted: {int(test_predict_inv[i, 0])}")

print("\n✅ LSTM forecasting finished. Plot saved as 'lstm_forecast.png'.")
