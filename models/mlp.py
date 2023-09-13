from models.model import Model
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from utils.technical_indicators import TechnicalIndicators


class MLP(Model):
    def __init__(self, name: str, training_set_percentage=0.8) -> None:
        super().__init__(name)
        self.training_set_percentage = training_set_percentage
        self.mlp, self.scaler = None, None

    def train(self, df: pd.DataFrame) -> None:
        # Create formatted training data for the MLP and separate it into training and validation sets
        print(f'Formatting MLP training data for {self.name}...')

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
        with open(f'./models/model_files/{self.name}_scaler.pickle', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Try different hyperparameters
        all_combos = []

        for hidden_layer_sizes in [(10), (25), (50), (100), (25, 50), (50, 100), (25, 25), (50, 50), (100, 100),
                                   (25, 50, 25), (100, 150, 100)]:
            for activation in ['tanh', 'relu', 'logistic']:
                for solver in ['sgd', 'adam']:
                    all_combos.append((hidden_layer_sizes, activation, solver))

        percentage_to_try = 0.25
        n_runs = int(percentage_to_try * len(all_combos))
        combos_to_try = random.sample(all_combos, n_runs)  # Perform a random search of the best hyperparameters
        best_validation_mse = np.inf

        # Train the MLP and choose the model with the best validation loss
        print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)
        print(f'Num training runs for {self.name}: {n_runs}')

        for hidden_layer_sizes, activation, solver in combos_to_try:
            curr_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver)
            curr_model.fit(x_train, y_train)

            y_validation_pred = curr_model.predict(x_validation)
            validation_mse = np.square(y_validation_pred - y_validation).mean()

            if validation_mse < best_validation_mse:
                print(f'Best validation MSE improved from {best_validation_mse} to {validation_mse}')
                best_validation_mse, self.mlp = validation_mse, curr_model

                with open(f'./models/model_files/{self.name}_mlp.pickle', 'wb') as f:
                    pickle.dump(self.mlp, f)

            n_runs -= 1
            print(f'Remaining runs: {n_runs} -- Best validation MSE so far: {best_validation_mse}')

        # Save the best MLP
        with open(f'./models/model_files/{self.name}_mlp.pickle', 'wb') as f:
            pickle.dump(self.mlp, f)

