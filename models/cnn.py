import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from models.model import Model
from pyts.image import GramianAngularField
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from utils.technical_indicators import TechnicalIndicators


class CNN(Model):
    def __init__(self, name: str, training_set_percentage=0.8, lookback=100) -> None:
        super().__init__(name)
        self.training_set_percentage = training_set_percentage
        self.lookback = lookback
        self.scaler = None

    def _convert_to_image_data(self, data_slice: np.array) -> np.array:
        gasf_transformer = GramianAngularField(method='summation')
        gasf_subset = gasf_transformer.transform(data_slice)

        return gasf_subset

    def load_model(self) -> None:
        self.cnn = load_model(f'../models/model_files/{self.name}_cnn')
        self.scaler = pickle.load(open(f'../models/model_files/{self.name}_scaler.pickle', 'rb'))

    def predict(self, x: np.array) -> Tuple[float, float, float, float]:
        x_scaled = self.scaler.transform(x)
        x_image = self._convert_to_image_data(x_scaled)

        return self.cnn.predict(x_image.reshape(-1, self.lookback, x_image.shape[-1], x_image.shape[-1]))[0]

    def train(self, df: pd.DataFrame) -> None:
        # Create formatted training data for the CNN and separate it into training and validation sets
        print(f'Formatting CNN training data for {self.name}...')

        df_train = TechnicalIndicators.format_data_for_ml_model(df)
        labels_df = df_train[['bid_pips_down', 'bid_pips_up', 'ask_pips_down', 'ask_pips_up']]
        assert len(df_train) == len(labels_df)

        self.scaler = StandardScaler()
        df_train = self.scaler.fit_transform(df_train)

        cnn_training_data = []

        for i in range(self.lookback, len(df_train)):
            df_slice = df_train[i - self.lookback:i, :]
            slice_image = self._convert_to_image_data(df_slice)
            bid_pips_down, bid_pips_up, ask_pips_down, ask_pips_up = labels_df.iloc[i, :]

            cnn_training_data.append((slice_image, np.array(
                [bid_pips_down, bid_pips_up, ask_pips_down, ask_pips_up])))

        np.random.shuffle(cnn_training_data)

        train_cutoff_index = int(len(cnn_training_data) * self.training_set_percentage)
        train_set, validation_set = cnn_training_data[:train_cutoff_index], \
                                    cnn_training_data[train_cutoff_index:]

        x_train, y_train, x_validation, y_validation = [], [], [], []

        for seq, target in train_set:
            x_train.append(seq)
            y_train.append(target)

        for seq, target in validation_set:
            x_validation.append(seq)
            y_validation.append(target)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_validation = np.array(x_validation)
        y_validation = np.array(y_validation)

        # Save the scaler
        with open(f'../models/model_files/{self.name}_scaler.pickle', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Create and train the CNN
        print(f'Training CNN for {self.name}...')
        print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)

        cnn = Sequential()

        # Block 1
        cnn.add(
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
        cnn.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
        cnn.add(MaxPool2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.1))

        # Block 2
        cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        cnn.add(MaxPool2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.1))

        # Block 3
        cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        cnn.add(MaxPool2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.1))

        # Output/final block
        cnn.add(Flatten())
        cnn.add(Dense(128, activation='relu'))
        cnn.add(Dense(32, activation='relu'))
        cnn.add(Dense(8, activation='relu'))
        cnn.add(Dense(4, activation='relu'))

        n_epochs = 50
        batch_size = 32
        optimizer = Adam()
        cnn_file_path = f'../models/model_files/{self.name}_cnn'
        early_stop = EarlyStopping(
            monitor='val_mean_squared_error', verbose=1, patience=int(n_epochs * 0.1))
        model_checkpoint = ModelCheckpoint(
            cnn_file_path, monitor='val_mean_squared_error', save_best_only=True, verbose=1)
        cnn.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

        cnn.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_validation, y_validation),
            callbacks=[early_stop, model_checkpoint])
