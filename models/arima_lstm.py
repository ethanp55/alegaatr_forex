import pickle
import numpy as np
import pandas as pd
import pmdarima as pm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from typing import Tuple
from models.model import Model


class ArimaLSTM(Model):
    def __init__(self, name: str, arima_train_percentage=0.5, lstm_training_set_percentage=0.8, lookback=100) -> None:
        super().__init__(name)
        self.arima_train_percentage = arima_train_percentage
        self.lstm_training_set_percentage = lstm_training_set_percentage
        self.lookback = lookback
        self.arima_bid_pips_down, self.arima_bid_pips_up, self.arima_ask_pips_down, self.arima_ask_pips_up = \
            None, None, None, None

    def load_model(self) -> None:
        self.lstm = load_model(f'../models/model_files/{self.name}_lstm')
        self.arima_bid_pips_down = pickle.load(
            open(f'../models/model_files/{self.name}_arima_bid_pips_down.pickle', 'rb'))
        self.arima_bid_pips_up = pickle.load(open(f'../models/model_files/{self.name}_arima_bid_pips_up.pickle', 'rb'))
        self.arima_ask_pips_down = pickle.load(
            open(f'../models/model_files/{self.name}_arima_ask_pips_down.pickle', 'rb'))
        self.arima_ask_pips_up = pickle.load(open(f'../models/model_files/{self.name}_arima_ask_pips_up.pickle', 'rb'))

    def predict(self, n_periods: int, bid_pips_down: pd.DataFrame, bid_pips_up: pd.DataFrame,
                ask_pips_down: pd.DataFrame, ask_pips_up: pd.DataFrame) -> Tuple[float, float, float, float]:
        arima_bid_pips_down_pred = self.arima_bid_pips_down.predict(n_periods=n_periods).iloc[-self.lookback:, ]
        arima_bid_pips_up_pred = self.arima_bid_pips_up.predict(n_periods=n_periods).iloc[-self.lookback:, ]
        arima_ask_pips_down_pred = self.arima_ask_pips_down.predict(n_periods=n_periods).iloc[-self.lookback:, ]
        arima_ask_pips_up_pred = self.arima_ask_pips_up.predict(n_periods=n_periods).iloc[-self.lookback:, ]

        arima_bid_pips_down_errors = bid_pips_down.reset_index(drop=True) - arima_bid_pips_down_pred.reset_index(
            drop=True)
        arima_bid_pips_up_errors = bid_pips_up.reset_index(drop=True) - arima_bid_pips_up_pred.reset_index(drop=True)
        arima_ask_pips_down_errors = ask_pips_down.reset_index(drop=True) - arima_ask_pips_down_pred.reset_index(
            drop=True)
        arima_ask_pips_up_errors = ask_pips_up.reset_index(drop=True) - arima_ask_pips_up_pred.reset_index(drop=True)

        arima_errors = pd.concat([arima_bid_pips_down_errors, arima_bid_pips_up_errors, arima_ask_pips_down_errors,
                                  arima_ask_pips_up_errors], axis=1)

        return self.lstm.predict(np.array(arima_errors).reshape(-1, self.lookback, arima_errors.shape[-1]))[0]

    def train(self, df: pd.DataFrame) -> None:
        # Create a training set for the ARIMA model and a training set for the LSTM
        arima_train_cutoff_index = int(
            len(df) * self.arima_train_percentage)

        df_train_arima, df_train_lstm = df.iloc[:arima_train_cutoff_index, :], df.iloc[arima_train_cutoff_index:, :]
        df_train_arima.reset_index(drop=True, inplace=True)
        df_train_lstm.reset_index(drop=True, inplace=True)

        # Fit an ARIMA model for each time series we're trying to predict
        print(f'Fitting ARIMA models for {self.name}...')

        self.arima_bid_pips_down = pm.auto_arima(
            df_train_arima['bid_pips_down'], stepwise=False, seasonal=False)
        self.arima_bid_pips_up = pm.auto_arima(
            df_train_arima['bid_pips_up'], stepwise=False, seasonal=False)
        self.arima_ask_pips_down = pm.auto_arima(
            df_train_arima['ask_pips_down'], stepwise=False, seasonal=False)
        self.arima_ask_pips_up = pm.auto_arima(
            df_train_arima['ask_pips_up'], stepwise=False, seasonal=False)

        # Save the ARIMA info
        with open(f'../models/model_files/{self.name}_arima_bid_pips_down.pickle', 'wb') as f:
            pickle.dump(self.arima_bid_pips_down, f)

        with open(f'../models/model_files/{self.name}_arima_bid_pips_up.pickle', 'wb') as f:
            pickle.dump(self.arima_bid_pips_up, f)

        with open(f'../models/model_files/{self.name}_arima_ask_pips_down.pickle', 'wb') as f:
            pickle.dump(self.arima_ask_pips_down, f)

        with open(f'../models/model_files/{self.name}_arima_ask_pips_up.pickle', 'wb') as f:
            pickle.dump(self.arima_ask_pips_up, f)

        num_lstm_data_points = len(df_train_lstm)

        # Make predictions for each ARIMA model (will be used to train the LSTM)
        arima_bid_pips_down_pred = self.arima_bid_pips_down.predict(
            n_periods=num_lstm_data_points)
        arima_bid_pips_up_pred = self.arima_bid_pips_up.predict(
            n_periods=num_lstm_data_points)
        arima_ask_pips_down_pred = self.arima_ask_pips_down.predict(
            n_periods=num_lstm_data_points)
        arima_ask_pips_up_pred = self.arima_ask_pips_up.predict(
            n_periods=num_lstm_data_points)

        # Calculate the ARIMA errors
        arima_bid_pips_down_errors = df_train_lstm['bid_pips_down'] - arima_bid_pips_down_pred.reset_index(drop=True)
        arima_bid_pips_up_errors = df_train_lstm['bid_pips_up'] - arima_bid_pips_up_pred.reset_index(drop=True)
        arima_ask_pips_down_errors = df_train_lstm['ask_pips_down'] - arima_ask_pips_down_pred.reset_index(drop=True)
        arima_ask_pips_up_errors = df_train_lstm['ask_pips_up'] - arima_ask_pips_up_pred.reset_index(drop=True)

        assert len(arima_bid_pips_down_errors) == len(arima_bid_pips_up_errors) == len(
            arima_ask_pips_down_errors) == len(arima_ask_pips_up_errors) == num_lstm_data_points

        df_arima_errors = pd.concat([arima_bid_pips_down_errors, arima_bid_pips_up_errors,
                                     arima_ask_pips_down_errors, arima_ask_pips_up_errors], axis=1)

        # Create formatted training data for the LSTM and separate it into training and validation sets
        print(f'Formatting LSTM training data for {self.name}...')
        lstm_training_data = []

        for i in range(self.lookback, num_lstm_data_points):
            errors_slice = df_arima_errors.iloc[i - self.lookback:i, :]
            bid_pips_down, bid_pips_up, ask_pips_down, ask_pips_up = df_train_lstm.loc[df_train_lstm.index[i], [
                'bid_pips_down', 'bid_pips_up', 'ask_pips_down', 'ask_pips_up']]

            lstm_training_data.append((np.array(errors_slice), np.array(
                [bid_pips_down, bid_pips_up, ask_pips_down, ask_pips_up])))

        np.random.shuffle(lstm_training_data)

        lstm_train_cutoff_index = int(len(lstm_training_data) * self.lstm_training_set_percentage)
        lstm_train_set, lstm_validation_set = lstm_training_data[:lstm_train_cutoff_index], \
                                              lstm_training_data[lstm_train_cutoff_index:]

        x_train, y_train, x_validation, y_validation = [], [], [], []

        for seq, target in lstm_train_set:
            x_train.append(seq)
            y_train.append(target)

        for seq, target in lstm_validation_set:
            x_validation.append(seq)
            y_validation.append(target)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_validation = np.array(x_validation)
        y_validation = np.array(y_validation)

        # Create and train the LSTM
        print(f'Training LSTM for {self.name}...')
        print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)

        lstm = Sequential()

        lstm.add(LSTM(128, return_sequences=True))
        lstm.add(Dropout(0.2))
        lstm.add(BatchNormalization())

        lstm.add(LSTM(128, return_sequences=True))
        lstm.add(Dropout(0.2))
        lstm.add(BatchNormalization())

        lstm.add(LSTM(128))
        lstm.add(Dropout(0.2))
        lstm.add(BatchNormalization())

        lstm.add(Dense(128, activation='relu'))
        lstm.add(Dropout(0.2))

        lstm.add(Dense(4, activation='relu'))

        n_epochs = 100
        batch_size = 32
        optimizer = Adam()
        lstm_file_path = f'../models/model_files/{self.name}_lstm'
        early_stop = EarlyStopping(
            monitor='val_mean_squared_error', verbose=1, patience=int(n_epochs * 0.1))
        model_checkpoint = ModelCheckpoint(
            lstm_file_path, monitor='val_mean_squared_error', save_best_only=True, verbose=1)
        lstm.compile(loss='mean_squared_error', optimizer=optimizer,
                     metrics=['mean_squared_error'])

        lstm.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_validation, y_validation),
            callbacks=[early_stop, model_checkpoint])
