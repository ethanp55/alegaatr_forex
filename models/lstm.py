import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from models.model import Model
from sklearn.preprocessing import StandardScaler
from utils.technical_indicators import TechnicalIndicators


class Lstm(Model):
    def __init__(self, name: str, lstm_training_set_percentage=0.8, lookback=100) -> None:
        super().__init__(name)
        self.lstm_training_set_percentage = lstm_training_set_percentage
        self.lookback = lookback
        self.scaler = None

    def train(self, df: pd.DataFrame) -> None:
        # Create formatted training data for the LSTM and separate it into training and validation sets
        print(f'Formatting LSTM training data for {self.name}...')

        df_train = TechnicalIndicators.format_data_for_ml_model(df)
        labels_df = df_train[['bid_pips_down', 'bid_pips_up', 'ask_pips_down', 'ask_pips_up']]
        assert len(df_train) == len(labels_df)

        self.scaler = StandardScaler()
        df_train = self.scaler.fit_transform(df_train)

        lstm_training_data = []

        for i in range(self.lookback, len(df_train)):
            df_slice = df_train[i - self.lookback:i, :]
            bid_pips_down, bid_pips_up, ask_pips_down, ask_pips_up = labels_df.iloc[i, :]

            lstm_training_data.append((np.array(df_slice), np.array(
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

        # Save the scaler
        with open(f'./models/model_files/{self.name}_scaler.pickle', 'wb') as f:
            pickle.dump(self.scaler, f)

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
        lstm_file_path = f'./models/model_files/{self.name}_lstm'
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
