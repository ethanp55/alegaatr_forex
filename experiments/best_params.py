from strategies.alegaatr import AlegAATr
from strategies.bar_movement import BarMovement
from strategies.beep_boop import BeepBoop
from strategies.bollinger_bands import BollingerBands
from strategies.choc import Choc
from strategies.cnn import CNNStrategy
from strategies.ensemble import Ensemble
from strategies.keltner_channels import KeltnerChannels
from strategies.knn import KNNStrategy
from strategies.lstm import LstmStrategy
from strategies.ma_crossover import MACrossover
from strategies.macd import MACD
from strategies.macd_key_level import MACDKeyLevel
from strategies.macd_stochastic import MACDStochastic
from strategies.mlp import MLPStrategy
from strategies.psar import PSAR
from strategies.random_forest import RandomForestStrategy
from strategies.rsi import RSI
from strategies.squeeze_pro import SqueezePro
from strategies.stochastic import Stochastic
from strategies.supertrend import Supertrend
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def print_best_params() -> None:
    # strategies = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(), MACD(),
    #               MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(), Supertrend(), PSAR(),
    #               RSI(), Stochastic(), Supertrend(), BeepBoop(), Ensemble(), AlegAATr()]
    strategies = [AlegAATr()]

    for currency_pair in CURRENCY_PAIRS:
        for time_frame in TIME_FRAMES:
            for year in YEARS[1:-1]:
                pair_time_frame_str = f'{currency_pair}_{time_frame}'
                cnn_model_name = f'CNN_{pair_time_frame_str}'
                knn_model_name = f'KNN_{pair_time_frame_str}'
                lstm_model_name = f'LSTM_{pair_time_frame_str}'
                mlp_model_name = f'MLP_{pair_time_frame_str}'
                rf_model_name = f'RandomForest_{pair_time_frame_str}'

                # ml_strategies = [CNNStrategy(cnn_model_name), KNNStrategy(knn_model_name), LstmStrategy(lstm_model_name),
                #                  MLPStrategy(mlp_model_name), RandomForestStrategy(rf_model_name)]
                ml_strategies = []

                all_strategies = strategies + ml_strategies

                print(f'{pair_time_frame_str}')

                for strategy in all_strategies:
                    try:
                        strategy.load_best_parameters(currency_pair, time_frame, year)
                        print(f'{strategy.name}')
                        strategy.print_parameters()

                    except:
                        pass
    
                print()


if __name__ == "__main__":
    print_best_params()
