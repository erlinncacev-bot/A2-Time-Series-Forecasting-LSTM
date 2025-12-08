# A2-Time-Series-Forecasting-LSTM
Time series forecasting using LSTM on the International Airline Passengers dataset with TensorFlow Keras.

https://colab.research.google.com/drive/1HqJFr4hOpC4PV2mP3pu882qRaTt-EUGe

VIERNES, Erlinn CACE T. 
BSCS-3B


SUMMARY

This project focuses on applying Long Short-Term Memory (LSTM) neural networks to perform time series forecasting using the International Airline Passengers Dataset. The main objective is to predict future airline passenger counts based on historical monthly data using deep learning techniques implemented in TensorFlow Keras.

The dataset contains monthly records of airline passengers from 1949 to 1960, making it a classic benchmark dataset for time series analysis. Since time series data has dependencies over time, the project uses an LSTM model, which is specifically designed to handle long-term sequence patterns.


OBJECTIVES

To understand time series forecasting using deep learning

To implement LSTM using TensorFlow Keras

To train a model that predicts future airline passenger values

To evaluate model performance using loss values

To visualize the predicted versus actual data



METHODOLOGY

1. Dataset Loading

The airline passenger dataset is loaded and inspected.

The data consists of two columns: Month and Passengers.



2. Data Preprocessing

The dataset is normalized for better training performance.

The data is split into training and testing sets.

Sequences are created using time-step windows for LSTM input.



3. Model Architecture

The model uses two LSTM layers and one Dense output layer.

LSTM layers extract time-dependent features.

The Dense layer outputs the final predicted value.



4. Model Training

The model is trained for 30 epochs.

Loss function: Mean Squared Error (MSE)

Optimizer: Adam



5. Prediction and Evaluation

The trained model predicts future passenger values.

Predictions are compared with the actual passenger data.

Model performance is evaluated using validation loss.



RESULTS AND OBSERVATIONS

The LSTM model successfully learned the trend and seasonality of the dataset.

Training and validation loss values decreased consistently.

The predicted values closely matched the actual passenger counts.

The model demonstrated strong capability in handling sequential time series data.



CONCLUSIONS

This project proves that LSTM neural networks are highly effective for time series forecasting problems. By training on historical airline passenger data, the model successfully predicts future passenger trends with a low error rate. This system can be extended to real-world forecasting applications such as finance, transportation, and business analytics.
