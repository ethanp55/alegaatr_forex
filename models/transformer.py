import keras
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, Layer, LayerNormalization, \
    MultiHeadAttention
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.optimizers import Adam
from models.model import Model as CustomModel
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
from utils.technical_indicators import TechnicalIndicators


@keras.saving.register_keras_serializable()
class Time2Vec(Layer):
    def __init__(self, linear_size: int = 1, n_components: int = 1) -> None:
        super(Time2Vec, self).__init__()

        # Parameters
        self.linear_size = linear_size
        self.n_components = n_components

        # Layers
        self.linear = Dense(self.linear_size)
        self.w = self.add_weight((self.n_components, 1), initializer='random_normal')  # Weights
        self.b = self.add_weight((self.n_components, 1), initializer='zeros')  # Biases

    def get_config(self) -> Dict:
        return {
            'linear_size': self.linear_size,
            'n_components': self.n_components
        }

    def call(self, inputs: np.array) -> tf.Tensor:
        linear_output = self.linear(inputs)

        periodic_output = []
        for i in range(self.n_components):
            periodic_output.append(tf.sin(self.w[i] * inputs + self.b[i]))
            periodic_output.append(tf.cos(self.w[i] * inputs + self.b[i]))

        time2vec_output = tf.concat([linear_output] + periodic_output, axis=-1)
        return time2vec_output


@keras.saving.register_keras_serializable()
class TransformerBlock(Layer):
    def __init__(self, n_heads: int = 12, dense_dim: int = 256, dropout_rate: float = 0.1):
        super(TransformerBlock, self).__init__()

        # Parameters
        self.n_heads = n_heads
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate

        # Layers
        self.mh_attention = MultiHeadAttention(num_heads=self.n_heads, key_dim=self.dense_dim)
        self.dense_1 = Dense(self.dense_dim)
        self.dense_2 = Dense(self.dense_dim)
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.dropout_1 = Dropout(self.dropout_rate)
        self.dropout_2 = Dropout(self.dropout_rate)

    def get_config(self) -> Dict:
        return {
            'n_heads': self.n_heads,
            'dense_dim': self.dense_dim,
            'dropout_rate': self.dropout_rate
        }

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        # Multi-headed attention
        x = self.mh_attention(inputs, inputs)
        x = self.dropout_1(x, training=training)
        x = x + inputs
        x = self.layer_norm_1(x)

        # Dense/feed-forward
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=training)
        x = self.layer_norm_2(x)

        return x


class TransformerNetwork(Model):
    def __init__(self, linear_size: int = 1, n_components: int = 1, n_heads: int = 12, dense_dim: int = 256,
                 dropout_rate: float = 0.1, penultimate_dense_dim: int = 64) -> None:
        super(TransformerNetwork, self).__init__()

        # Parameters
        self.linear_size = linear_size
        self.n_components = n_components
        self.n_heads = n_heads
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
        self.penultimate_dense_dim = penultimate_dense_dim

        # Layers
        self.day_time_to_vec = Time2Vec(linear_size=self.linear_size, n_components=self.n_components)
        self.hour_time_to_vec = Time2Vec(linear_size=self.linear_size, n_components=self.n_components)
        self.minute_time_to_vec = Time2Vec(linear_size=self.linear_size, n_components=self.n_components)
        self.t_block_1 = TransformerBlock(n_heads=self.n_heads, dense_dim=self.dense_dim,
                                          dropout_rate=self.dropout_rate)
        self.t_block_2 = TransformerBlock(n_heads=self.n_heads, dense_dim=self.dense_dim,
                                          dropout_rate=self.dropout_rate)
        self.t_block_3 = TransformerBlock(n_heads=self.n_heads, dense_dim=self.dense_dim,
                                          dropout_rate=self.dropout_rate)
        self.pooling = GlobalAveragePooling1D()
        self.dropout_1 = Dropout(self.dropout_rate)
        self.dropout_2 = Dropout(self.dropout_rate)
        self.dense = Dense(self.penultimate_dense_dim)
        self.output_layer = Dense(4)

    def get_config(self) -> Dict:
        return {
            'linear_size': self.linear_size,
            'n_components': self.n_components,
            'n_heads': self.n_heads,
            'dense_dim': self.dense_dim,
            'dropout_rate': self.dropout_rate,
            'penultimate_dense_dim': self.penultimate_dense_dim
        }

    def call(self, inputs: np.array, training: bool = False) -> tf.Tensor:
        # Extract the time values (assumes they are the last 3 features)
        days, hours, minutes = inputs[:, :, -3], inputs[:, :, -2], inputs[:, :, -1]
        n_rows = inputs.shape[1]
        days = tf.reshape(days, [-1, n_rows, 1])
        hours = tf.reshape(hours, [-1, n_rows, 1])
        minutes = tf.reshape(minutes, [-1, n_rows, 1])

        # Time embeddings
        days_embedding = self.day_time_to_vec(days)
        hours_embedding = self.hour_time_to_vec(hours)
        minutes_embedding = self.minute_time_to_vec(minutes)

        # Attach time embeddings back to the original data
        x = tf.concat([inputs, days_embedding, hours_embedding, minutes_embedding], axis=-1)

        # Pass through layers
        x = self.t_block_1(x, training=training)
        x = self.t_block_2(x, training=training)
        x = self.t_block_3(x, training=training)
        x = self.pooling(x)
        x = self.dropout_1(x, training=training)
        x = self.dense(x)
        x = self.dropout_2(x, training=training)
        x = self.output_layer(x)

        return x


class TransformerModel(CustomModel):
    def __init__(self, name: str, training_set_percentage=0.8, lookback=100) -> None:
        super().__init__(name)
        self.training_set_percentage = training_set_percentage
        self.lookback = lookback
        self.scaler = None

    def load_model(self) -> None:
        self.transformer = load_model(f'../models/model_files/{self.name}_transformer.keras')
        self.scaler = pickle.load(open(f'../models/model_files/{self.name}_scaler.pickle', 'rb'))

    def predict(self, x: np.array) -> Tuple[float, float, float, float]:
        x_scaled = self.scaler.transform(x)

        return self.transformer.predict(x_scaled.reshape(-1, self.lookback, x_scaled.shape[-1]), verbose=0)[0]

    def train(self, df: pd.DataFrame) -> None:
        # Create formatted training data for the transformer and separate it into training and validation sets
        print(f'Formatting transformer training data for {self.name}...')

        df_train = TechnicalIndicators.format_data_for_transformer(df)
        labels_df = df_train[['bid_pips_down', 'bid_pips_up', 'ask_pips_down', 'ask_pips_up']]
        assert len(df_train) == len(labels_df)

        self.scaler = StandardScaler()
        df_train = self.scaler.fit_transform(df_train)

        transformer_training_data = []

        for i in range(self.lookback, len(df_train)):
            df_slice = df_train[i - self.lookback:i, :]
            bid_pips_down, bid_pips_up, ask_pips_down, ask_pips_up = labels_df.iloc[i, :]

            transformer_training_data.append((df_slice, np.array(
                [bid_pips_down, bid_pips_up, ask_pips_down, ask_pips_up])))

        np.random.shuffle(transformer_training_data)

        train_cutoff_index = int(len(transformer_training_data) * self.training_set_percentage)
        train_set, validation_set = transformer_training_data[:train_cutoff_index], \
                                    transformer_training_data[train_cutoff_index:]

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

        # Create and train the transformer
        print(f'Training transformer for {self.name}...')
        print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)

        transformer = TransformerNetwork()
        n_epochs = 50
        early_stop = int(n_epochs * 0.1)
        n_epochs_without_change = 0
        best_val_mse = np.inf
        val_metric = MeanSquaredErrorMetric()
        batch_size = 32
        loss_fn = MeanSquaredError()
        optimizer = Adam()
        transformer_file_path = f'../models/model_files/{self.name}_transformer.keras'
        n_batches, n_val_batches = len(x_train) // batch_size, len(x_validation) // batch_size

        for epoch in range(n_epochs):
            print(f'Epoch {epoch + 1}')

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                curr_x, curr_y = x_train[start_idx:end_idx, :, :], y_train[start_idx:end_idx, :]

                with tf.GradientTape() as tape:
                    predictions = transformer(curr_x, training=True)
                    loss = loss_fn(curr_y, predictions)

                grads = tape.gradient(loss, transformer.trainable_weights)
                optimizer.apply_gradients(zip(grads, transformer.trainable_weights))

            # Once all batches for the epoch are complete, calculate the validation loss
            for i in range(n_val_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                curr_x, curr_y = x_validation[start_idx:end_idx, :, :], y_validation[start_idx:end_idx, :]
                val_predictions = transformer(curr_x)
                val_metric.update_state(curr_y, val_predictions)

            val_mse = val_metric.result()
            val_metric.reset_state()

            print(f'Train MSE = {loss}, Validation MSE = {val_mse}')

            # Make updates if the validation performance improved
            if val_mse < best_val_mse:
                print(f'Validation performance improved from {best_val_mse} to {val_mse}')

                # Reset the number of epochs that have passed without any improvement/change
                n_epochs_without_change = 0

                # Update the best validation performance metric
                best_val_mse = val_mse

                # Save the network
                transformer.save(transformer_file_path)

            else:
                # Increment the number of epochs that have passed without any improvement/change
                n_epochs_without_change += 1

                # If sufficient epochs have passed without improvement, cancel the training process
                if n_epochs_without_change >= early_stop:
                    print(f'EARLY STOPPING - {n_epochs_without_change} HAVE PASSED WITHOUT VALIDATION IMPROVEMENT')
                    break

            # Shuffle the training data at the end of each epoch
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            x_train = x_train[indices, :, :]
            y_train = y_train[indices, :]
