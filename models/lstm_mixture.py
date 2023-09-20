from models.model import Model
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Model as TfModel
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Layer, LSTM, Multiply, Softmax
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adam
from typing import Optional, List, Tuple
from utils.technical_indicators import TechnicalIndicators


class LstmMixtureLayer(Layer):
    def __init__(self, activation_function: str) -> None:
        super(LstmMixtureLayer, self).__init__()

        self.lstm1 = LSTM(128, return_sequences=True)
        self.dropout1 = Dropout(0.2)
        self.batch_norm1 = BatchNormalization()

        self.lstm2 = LSTM(128, return_sequences=True)
        self.dropout2 = Dropout(0.2)
        self.batch_norm2 = BatchNormalization()

        self.lstm3 = LSTM(128)
        self.dropout3 = Dropout(0.2)
        self.batch_norm3 = BatchNormalization()

        self.dense1 = Dense(128, activation=activation_function)
        self.dropout4 = Dropout(0.2)

        self.dense2 = Dense(4, activation=activation_function)

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        x = self.lstm1(input_tensor)
        x = self.dropout1(x)
        x = self.batch_norm1(x)

        x = self.lstm2(x)
        x = self.dropout2(x)
        x = self.batch_norm2(x)

        x = self.lstm3(x)
        x = self.dropout3(x)
        x = self.batch_norm3(x)

        x = self.dense1(x)
        x = self.dropout3(x)

        x = tf.expand_dims(x, axis=1)  # This extra dimension represent the output from each LSTM layer

        return self.dense2(x)


class LstmMixtureNetwork(TfModel):
    def __init__(self, activation_functions: Optional[List[str]] = None) -> None:
        super(LstmMixtureNetwork, self).__init__()
        activations = activation_functions if activation_functions is not None else \
            ['linear', 'relu', 'sigmoid', 'tanh', 'swish']

        # LSTM layers, each with a different activation function
        self.lstm_layers = [LstmMixtureLayer(activation_function) for activation_function in activations]

        # Softmax layer
        self.softmax = Softmax()

        # Multiplication layer
        self.multiplication = Multiply()

        # Flatten layer
        self.flatten = Flatten()

        # Output/final layer
        self.out = Dense(4, activation='linear')

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        lstm_layer_outputs = [lstm_layer(input_tensor) for lstm_layer in self.lstm_layers]
        lstm_outputs = tf.concat(lstm_layer_outputs, axis=1)

        x = self.softmax(lstm_outputs)

        x = self.multiplication([lstm_outputs, x])

        x = self.flatten(x)

        return self.out(x)


class LstmMixture(Model):
    def __init__(self, name: str, lstm_training_set_percentage=0.8, lookback=100) -> None:
        super().__init__(name)
        self.lstm_training_set_percentage = lstm_training_set_percentage
        self.lookback = lookback
        self.scaler = None

    def load_model(self) -> None:
        self.mixture = load_model(f'../models/model_files/{self.name}_mixture')
        self.scaler = pickle.load(open(f'../models/model_files/{self.name}_scaler.pickle', 'rb'))

    def predict(self, x: np.array) -> Tuple[float, float, float, float]:
        x_scaled = self.scaler.transform(x)

        return self.mixture.predict(x_scaled.reshape(-1, self.lookback, x_scaled.shape[-1]))[0]

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
        with open(f'../models/model_files/{self.name}_scaler.pickle', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Items needed for training and testing, since we're running a custom TF model (optimizer, metrics, etc.)
        optimizer = Adam()
        loss_fn = MeanSquaredError()
        val_metric = MeanSquaredErrorMetric()
        best_val_mse = np.inf
        n_epochs_without_change = 0
        n_epochs = 100
        n_epochs_for_cancel = int(n_epochs * 0.1)
        batch_size = 32

        # Create and train the LSTM
        print(f'Training LSTM for {self.name}...')
        x_train, y_train = tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train)
        x_validation, y_validation = tf.convert_to_tensor(x_validation), tf.convert_to_tensor(y_validation)
        print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)

        lstm_mixture = LstmMixtureNetwork()

        for epoch in range(n_epochs):
            print(f'Epoch {epoch + 1}')

            for step, (x_batch, y_batch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    predictions = lstm_mixture(x_batch)
                    loss_value = loss_fn(y_batch, predictions)
                    loss_value = tf.reduce_mean(loss_value)

                grads = tape.gradient(loss_value, lstm_mixture.trainable_weights)
                optimizer.apply_gradients(zip(grads, lstm_mixture.trainable_weights))

            # Check validation loss for early stopping or updating the best model
            validation_predictions = lstm_mixture(x_validation)
            val_metric.update_state(y_validation, validation_predictions)
            val_mse = val_metric.result()
            val_metric.reset_states()

            print(f'Train MSE = {loss_value}, Validation MSE = {val_mse}')

            if val_mse < best_val_mse:
                print(f'Validation performance improved from {best_val_mse} to {val_mse}')

                # Reset the number of epochs that have passed without any improvement/change
                n_epochs_without_change = 0

                # Update the best validation performance metric
                best_val_mse = val_mse

                # Save the network
                save_model(lstm_mixture, f'/./models/model_files/{self.name}_mixture')

            else:
                # Increment the number of epochs that have passed without any improvement/change
                n_epochs_without_change += 1

                # If sufficient epochs have passed without improvement, cancel the training process
                if n_epochs_without_change >= n_epochs_for_cancel:
                    print(f'EARLY STOPPING - {n_epochs_without_change} HAVE PASSED WITHOUT VALIDATION IMPROVEMENT')
                    break
