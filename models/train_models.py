from data.data_loader import DataLoader
from models.cnn import CNN
from models.knn import KNN
from models.lstm import Lstm
from models.lstm_mixture import LstmMixture
from models.mlp import MLP
from models.random_forest import RandomForest
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES


def train_models() -> None:
    # We have data for different currency pairs and time frames (for comparison/test purposes); train each model on
    # each dataset
    for time_frame in TIME_FRAMES:
        for currency_pair in CURRENCY_PAIRS:
            # String that represents the combination of the current currency pair and time frame
            pair_time_frame_str = f'{currency_pair}_{time_frame}'

            # Multiplier to convert price value to pips - for example, 0.0050, when working with the EUR/USD pair, is
            # 50 pips (0.0050 * 10000 = 50)
            pips_multiplier = 100 if 'Jpy' in currency_pair else 10000

            # Load a training data frame for each currency pair amd time frame
            df_train = DataLoader.load_training_data(currency_pair, time_frame, pips_multiplier)

            # Models we want to train
            models = [Lstm(f'LSTM_{pair_time_frame_str}'),
                      LstmMixture(f'LstmMixture_{pair_time_frame_str}'),
                      CNN(f'CNN_{pair_time_frame_str}'),
                      RandomForest(f'RandomForest_{pair_time_frame_str}'),
                      KNN(f'KNN_{pair_time_frame_str}'),
                      MLP(f'MLP_{pair_time_frame_str}')]

            # Train each model
            for model in models:
                model.train(df_train)


if __name__ == "__main__":
    train_models()
