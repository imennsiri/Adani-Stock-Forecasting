# Adani-Stock-Forecasting
## 📦Libraries Used
python
Copy
Edit
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
import warnings  
from sklearn.metrics import mean_absolute_error  
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense  
from tqdm import tqdm  
import os  
from mplfinance.original_flavor import candlestick_ohlc  
import matplotlib.dates as mdates  
Data Handling: pandas, numpy

Visualization: matplotlib, seaborn, mplfinance

Deep Learning: TensorFlow (Keras)

Evaluation: mean_absolute_error

Utilities: tqdm, warnings, os

## 📁 Dataset
File: adani.csv

Columns:

timestamp, symbol, company

open, high, low, close, volume

dividends, stock_splits

Shape: 39,332 rows × 10 columns

Source: kaggle
https://www.kaggle.com/datasets/swaptr/adani-stocks/data

## 📊 Exploratory Data Analysis (EDA)
Summary statistics using df.describe()

Checked for missing values

Correlation heatmap of numerical columns

Distribution plots for open, high, low, close

Time series plots for each company's close prices

30-day moving average overlay

Candlestick charts using OHLC data

Volume vs Close price relationship

## 🛠 Feature Engineering
Added 7-day moving average for close price:

python
Copy
Edit
df['Moving_Avg_Close'] = df['close'].rolling(window=7).mean()
## 🧪 Data Preparation
Used only close price for univariate time series

Scaled data using MinMaxScaler (range: 0 to 1)

Created sequences with previous day as input, next day as label

Train-test split: 80% training, 20% testing

Reshaped input for LSTM: (samples, time_steps=1, features=1)

## 🧠 LSTM Model
python
Copy
Edit
model = Sequential()
model.add(LSTM(units=50, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
Architecture:

1 LSTM layer with 50 units

1 Dense layer with 1 unit

Loss function: Mean Squared Error

Optimizer: Adam

Training:

Batch size: 15

Epochs: 30

Trained using train_on_batch for precise control

## 🔮 Prediction
Used the trained model to predict values on the test set

Inverse-transformed predictions using scaler.inverse_transform

Saved predictions to predictions.csv

Visualized predictions vs actual values using matplotlib

## 📈 Results
Mean Absolute Error (MAE): ~3.27

Visual alignment between predicted and actual closing prices is good

Predictions saved for future reference

## 📁 Output Files
predictions.csv: Contains timestamp and Predicted_Close

## ✅ Future Improvements
Use multivariate time series (include volume, high-low, etc.)

Implement other models: ARIMA, GRU, XGBoost

Tune LSTM hyperparameters using GridSearch or Optuna

Use early stopping, dropout, or bidirectional LSTMs

Forecast multiple steps ahead (not just one day)

## 🧠 Final Notes
This project demonstrates the potential of LSTM models for financial time series forecasting. While it's a basic univariate setup, it opens the door to more complex applications involving multiple stocks, macroeconomic indicators, or sentiment analysis.
