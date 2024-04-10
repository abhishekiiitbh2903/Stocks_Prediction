# Model Building Ends Here 
import tensorflow as tf
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

def train_model():
    data = pd.read_csv('Google_train_data.csv')
    data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
    data = data.dropna()

    sc = MinMaxScaler(feature_range=(0, 1))
    closing_prices = data.iloc[:, 4:5].values
    closing_prices_scaled = sc.fit_transform(closing_prices)

    X_train = []
    y_train = []

    for i in range(60, len(data)):
        X_train.append(closing_prices_scaled[i-60:i, 0])
        y_train.append(closing_prices_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss=tf.compat.v1.losses.mean_squared_error)

    hist = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

    # Plotting the training loss
    # plt.figure(figsize=(8, 5))
    # plt.plot(hist.history['loss'])
    # plt.title('Training Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # st.pyplot(plt)

    return model,sc


model,sc=train_model()
# pickle.dump(model,open('model.pkl','wb'))
model.save('model.h5')
pickle.dump(sc,open('scaler.pkl','wb'))