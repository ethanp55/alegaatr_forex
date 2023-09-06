from data.data_loader import DataLoader
from models.arima_lstm import ArimaLSTM
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES


def train_models() -> None:
    # We have data for different currency pairs and time frames (for comparison/test purposes); train each model on each dataset
    for time_frame in TIME_FRAMES:
        for currency_pair in CURRENCY_PAIRS:
            # String that represents the combination of the current currency pair and time frame
            pair_time_frame_str = f'{currency_pair}_{time_frame}'

            # Multiplier to convert price value to pips - for example, 0.0050, when working with the EUR/USD pair, is
            # 50 pips (0.0050 * 10000 = 50)
            pips_multiplier = 100 if 'Jpy' in currency_pair else 10000

            # Load a training data frame for each currency pair and time frame
            data_file = f'./data/files/Oanda_{pair_time_frame_str}_2015-2023.csv'
            df_train = DataLoader.load_training_data(
                data_file, pips_multiplier)

            # Models we want to train
            models = [ArimaLSTM(f'ArimaLSTM_{pair_time_frame_str}')]

            # Train each model
            for model in models:
                model.train(df_train)


if __name__ == "__main__":
    train_models()
