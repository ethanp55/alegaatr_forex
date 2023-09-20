from models.model import Model
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple
from utils.technical_indicators import TechnicalIndicators


class RandomForest(Model):
    def __init__(self, name: str, training_set_percentage=0.8) -> None:
        super().__init__(name)
        self.training_set_percentage = training_set_percentage
        self.random_forest, self.scaler = None, None

    def load_model(self) -> None:
        self.random_forest = pickle.load(open(f'../models/model_files/{self.name}_rf.pickle', 'rb'))
        self.scaler = pickle.load(open(f'../models/model_files/{self.name}_scaler.pickle', 'rb'))

    def predict(self, x: np.array) -> Tuple[float, float, float, float]:
        x_scaled = self.scaler.transform(x)

        return self.random_forest.predict(x_scaled)[0]

    def train(self, df: pd.DataFrame) -> None:
        # Create formatted training data for the random forest and separate it into training and validation sets
        print(f'Formatting random forest training data for {self.name}...')

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

        for n_estimators in [5, 10, 15, 20, 25, 50]:
            for min_samples_leaf in [5, 10, 15, 20, 25, 50]:
                for max_depth in [3, 4, 5, 6, 7, 8, 9, 10]:
                    for min_samples_split in [2, 3, 4, 5, 10, 15]:
                        all_combos.append((n_estimators, min_samples_leaf, max_depth, min_samples_split))

        percentage_to_try = 0.1
        n_runs = int(percentage_to_try * len(all_combos))
        combos_to_try = random.sample(all_combos, n_runs)  # Perform a random search of the best hyperparameters
        best_validation_mse = np.inf

        # Train the random forest and choose the model with the best validation loss
        print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)
        print(f'Num training runs for {self.name}: {n_runs}')

        for n_estimators, min_samples_leaf, max_depth, min_samples_split in combos_to_try:
            curr_model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                               max_depth=max_depth, min_samples_split=min_samples_split)
            curr_model.fit(x_train, y_train)

            y_validation_pred = curr_model.predict(x_validation)
            validation_mse = np.square(y_validation_pred - y_validation).mean()

            if validation_mse < best_validation_mse:
                print(f'Best validation MSE improved from {best_validation_mse} to {validation_mse}')
                best_validation_mse, self.random_forest = validation_mse, curr_model

            n_runs -= 1
            print(f'Remaining runs: {n_runs} -- Best validation MSE so far: {best_validation_mse}')

        # Save the best random forest
        with open(f'../models/model_files/{self.name}_rf.pickle', 'wb') as f:
            pickle.dump(self.random_forest, f)

