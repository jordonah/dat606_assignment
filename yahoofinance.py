import pandas as pd
import numpy as np
import yfinance as yf  # For fetching data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from datetime import date# import talib
import sys
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class StockPredictor():

    def __init__(self, start_date, end_date):
        # initialize the start and end date variables as properties of the class
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self, ticker):
        # download the data using the start and end dates as a range, and then using a ticker passed to the script.
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date)

            close_prices = data['Close'].values.reshape(-1, 1)
            print(close_prices)
        except Exception as e:
            print("Error. The download of the data failed with the following error: ", e)
            sys.exit()
        self.dataset = data

    def preprocess_data(self):
        # clean and preprocess the data: start with identifying null values and seeing the dimensions of the data / columns. 
        print(self.dataset.isnull().sum())
        print(self.dataset.describe())
        print(self.dataset.info())
        print(self.dataset.head())
        print(self.dataset.columns)
        self.dataset.ffill(inplace=True)  # Forward-fill missing values
        self.dataset.interpolate(method='time', inplace=True)
        z_scores = (self.dataset['Close'] - self.dataset['Close'].mean()) / self.dataset['Close'].std()
        self.dataset = self.dataset[(np.abs(z_scores) < 3)]
        # identify the day of the week. This is needed, as no ticker history should occur on weekends. 
        self.dataset['Day_of_Week'] = self.dataset.index.dayofweek
        self.dataset['Month'] = self.dataset.index.month
        # check the information on the dataset again after our changes.
        print(self.dataset.info())
        print(self.dataset.head())
        # Use MinMaxSclaer to scale the numeric values. 
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.dataset[['Close', 'Volume']])
        self.train = self.dataset.loc[self.dataset.index < self.split_date]
        self.test = self.dataset.loc[self.dataset.index >= self.split_date]
        print(len(self.train))
        print(len(self.test))
        self.dataset['Close'].plot(title='Stock Closing Price')
        plt.show()
        # self.dataset.to_csv("processed_stock_data.csv")
        return self.train, self.test

    def convert_to_dates(self, percent):
        starting_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        ending_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        if starting_date > ending_date:
            print("Error: Start date must be before end date")

        # Calculate time difference
        delta = ending_date - starting_date
        fraction = percent / 100.0
        
        # Calculate target date
        result_date = starting_date + (delta * fraction)
        
        # Format and print result
        print(f"\nThe {percent}% date between {starting_date.date()} and {ending_date.date()} is:")
        print(f"-> {result_date.date()}")
        self.split_date = datetime.strftime(result_date, "%Y-%m-%d")

    def scale_data(self):
        # scale numeric columns and create sequences for deep learning analysis.
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(self.train)
        test_scaled = scaler.transform(self.test)

        def create_sequences(data, seq_length, target_col_idx):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])  # Input: (seq_length, n_features)
                y.append(data[i+seq_length, target_col_idx])  # Output: Next day's price
            return np.array(X), np.array(y)

        sequence_length = 60
        target_col_idx = 3  # Index of "Close" in the feature matrix

        self.X_train, self.y_train = create_sequences(train_scaled, sequence_length, target_col_idx)
        self.X_test, self.y_test = create_sequences(test_scaled, sequence_length, target_col_idx)

        model = Sequential()
        # First LSTM layer (return sequences=True for stacking)
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dropout(0.2))
        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        # Output layer (1 neuron for regression)
        model.add(Dense(units=1))

        model.summary()

        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(
        self.X_train, self.y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
        )
        predictions = model.predict(self.X_test)

        # Inverse-transform predictions and actual values
        def inverse_transform_predictions(scaler, data, preds, target_col_idx):
            dummy = np.zeros((len(preds), data.shape[1]))
            dummy[:, target_col_idx] = preds.flatten()
            return scaler.inverse_transform(dummy)[:, target_col_idx]

        inv_predictions = inverse_transform_predictions(scaler, test_scaled, predictions, target_col_idx)
        inv_actual = inverse_transform_predictions(scaler, test_scaled, self.y_test, target_col_idx)

        plt.figure(figsize=(12, 6))
        plt.plot(inv_actual, label='Actual Price')
        plt.plot(inv_predictions, label='Predicted Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()
        test_loss = model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss (MSE): {test_loss}")