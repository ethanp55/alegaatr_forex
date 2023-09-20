from models.model import Model
import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from utils.technical_indicators import TechnicalIndicators


class KNN(Model):
    def __init__(self, name: str, training_set_percentage=0.8) -> None:
        super().__init__(name)
        self.training_set_percentage = training_set_percentage
        self.knn, self.scaler = None, None

    def load_model(self) -> None:
        self.knn = pickle.load(open(f'../models/model_files/{self.name}_knn.pickle', 'rb'))
        self.scaler = pickle.load(open(f'../models/model_files/{self.name}_scaler.pickle', 'rb'))

    def predict(self, x: np.array) -> Tuple[float, float, float, float]:
        x_scaled = self.scaler.transform(x)

        return self.knn.predict(x_scaled)[0]

    def train(self, df: pd.DataFrame) -> None:
        # Create formatted training data for the KNN model and separate it into training and validation sets
        print(f'Formatting KNN training data for {self.name}...')

        df_train = TechnicalIndicators.format_data_for_ml_model(df)
        labels_df = df_train[['bid_pips_down', 'bid_pips_up', 'ask_pips_down', 'ask_pips_up']]
        assert len(df_train) == len(labels_df)

        self.scaler = StandardScaler()
        df_train = self.scaler.fit_transform(df_train)

        training_data = []

        for i in range(1, len(df_train)):
            bid_pips_down, bid_pips_up, ask_pips_down, ask_pips_up = labels_df.iloc[i, :]

            training_data.append((df_train[i - 1, :],
                                  np.array([bid_pips_down, bid_pips_up, ask_pips_down, ask_pips_up])))

        np.random.shuffle(training_data)

        train_cutoff_index = int(len(training_data) * self.training_set_percentage)
        train_set, validation_set = training_data[:train_cutoff_index], \
                                    training_data[train_cutoff_index:]

        x_train, y_train, x_validation, y_validation = [], [], [], []

        for seq, target in train_set:
            x_train.append(seq)
            y_train.append(target)

        for seq, target in validation_set:
            x_validation.append(seq)
            y_validation.append(target)

        x_train, y_train, x_validation, y_validation = \
            np.array(x_train), np.array(y_train), np.array(x_validation), np.array(y_validation)

        # Save the scaler
        with open(f'../models/model_files/{self.name}_scaler.pickle', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Try different hyperparameters
        all_combos = []

        for n_neighbors in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 41, 51, 61, 71, 81, 91, 101, 125]:
            for weights in ['uniform', 'distance']:
                all_combos.append((n_neighbors, weights))

        n_runs = int(len(all_combos))
        best_validation_mse = np.inf

        # Train the KNN model and choose the model with the best validation loss
        print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)
        print(f'Num training runs for {self.name}: {n_runs}')

        for n_neighbors, weights in all_combos:
            curr_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
            curr_model.fit(x_train, y_train)

            y_validation_pred = curr_model.predict(x_validation)
            validation_mse = np.square(y_validation_pred - y_validation).mean()

            if validation_mse < best_validation_mse:
                print(f'Best validation MSE improved from {best_validation_mse} to {validation_mse}')
                best_validation_mse, self.knn = validation_mse, curr_model

            n_runs -= 1
            print(f'Remaining runs: {n_runs} -- Best validation MSE so far: {best_validation_mse}')

        # Save the best KNN
        with open(f'../models/model_files/{self.name}_knn.pickle', 'wb') as f:
            pickle.dump(self.knn, f)
