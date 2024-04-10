import streamlit as st
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,Flatten,LSTM,Dropout
import joblib
import matplotlib.pyplot as plt
import pickle

def load_model_and_scaler():
    model_file = "model.h5"
    scaler_file = "scaler.pkl"

    if os.path.isfile(model_file) and os.path.isfile(scaler_file):
        # If the model and scaler files exist, load them
        model = load_model(model_file)
        sc = joblib.load(scaler_file)

    # else:
    #     # If the model or scaler files don't exist, train the model and save them
    #     model, sc = train_model()
    #     model.save(model_file)
    #     joblib.dump(sc, scaler_file)

    return model, sc

# def train_model():
#     data = pd.read_csv('Google_train_data.csv')
#     data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
#     data = data.dropna()

#     sc = MinMaxScaler(feature_range=(0, 1))
#     closing_prices = data.iloc[:, 4:5].values
#     closing_prices_scaled = sc.fit_transform(closing_prices)

#     X_train = []
#     y_train = []

#     for i in range(60, len(data)):
#         X_train.append(closing_prices_scaled[i-60:i, 0])
#         y_train.append(closing_prices_scaled[i, 0])

#     X_train, y_train = np.array(X_train), np.array(y_train)
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#     model = Sequential()
#     model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=100, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=100, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=100, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1))

#     optimizer = 'adam'
#     model.compile(optimizer=optimizer, loss="mean_squared_error")

#     hist = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

#     # Plotting the training loss
#     plt.figure(figsize=(8, 5))
#     plt.plot(hist.history['loss'])
#     plt.title('Training Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     st.pyplot(plt)

#     return model, sc

def predict_stock_price(model, sc, input_data):
    input_data_scaled = sc.transform(input_data)
    X_test = []

    for i in range(60, len(input_data_scaled)):
        X_test.append(input_data_scaled[i-60:i, 0])

    X_test = np.array(X_test)
    # st.text(X_test[0:2])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # X_test_reshaped = [[item] for sublist in X_test for item in sublist]
    # st.text(X_test_reshaped[0:2])
    # X_test= [[x] for sublist in X_test for x in sublist]
    # st.text(X_test)
    y_pred_scaled = model.predict(X_test)
    predicted_price = sc.inverse_transform(y_pred_scaled)

    return input_data[60:, 0], predicted_price.flatten()

def main():
    st.title("Stock Price Prediction App")

    # Load the model and scaler
    model, sc = load_model_and_scaler()

    # Load and preprocess the default test data
    test_data = pd.read_csv('Google_test_data.csv')
    test_data["Close"] = pd.to_numeric(test_data["Close"], errors='coerce')
    test_data = test_data.dropna()
    test_data = test_data.iloc[:, 4:5]

    # Predict stock prices
    input_closing = test_data.iloc[:, 0:].values
    actual_prices, predicted_prices = predict_stock_price(model, sc, input_closing)

    # Plotting actual vs predicted graph
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, label='Actual Stock Price', color='red')
    plt.plot(predicted_prices, label='Predicted Stock Price', color='green')
    plt.title('Actual vs Predicted Stock Price')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)

    # Plotting Google_train_data
    st.subheader("Google Train Data")
    st.line_chart(pd.DataFrame({
        'Close Price': pd.to_numeric(pd.read_csv('Google_train_data.csv')["Close"], errors='coerce')
    }))

    # Plotting Google_test_data
    st.subheader("Google Test Data")
    st.line_chart(pd.DataFrame({
        'Close Price': pd.to_numeric(test_data["Close"], errors='coerce')
    }))


if __name__ == '__main__':
    main()
