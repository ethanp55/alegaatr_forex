from functools import partial
from multiprocessing import Pool
from runner.simulation_runner import SimulationRunner
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
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES


def test_strategies() -> None:
    # List of all of the regular strategies
    strategies = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(), MACD(),
                  MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(), Supertrend(), PSAR(),
                  RSI(), Stochastic(), Supertrend(), BeepBoop(), Ensemble()]

    # List of the final results to output
    test_results = []

    for currency_pair in CURRENCY_PAIRS:
        for time_frame in TIME_FRAMES:
            pair_time_frame_str = f'{currency_pair}_{time_frame}'

            # Model names for the ML strategies (they need to load in pair-time-specific data)
            cnn_model_name = f'CNN_{pair_time_frame_str}'
            knn_model_name = f'KNN_{pair_time_frame_str}'
            lstm_model_name = f'LSTM_{pair_time_frame_str}'
            mlp_model_name = f'MLP_{pair_time_frame_str}'
            rf_model_name = f'RandomForest_{pair_time_frame_str}'

            # List of ML strategies
            ml_strategies = [CNNStrategy(cnn_model_name), KNNStrategy(knn_model_name), LstmStrategy(lstm_model_name),
                             MLPStrategy(mlp_model_name), RandomForestStrategy(rf_model_name)]

            # List of all the strategies
            all_strategies = strategies + ml_strategies

            # Creates a new process for each strategy
            pool = Pool(processes=len(all_strategies))
            results = pool.map(
                partial(SimulationRunner.run_simulation, currency_pair=currency_pair, time_frame=time_frame,
                        optimize=False), all_strategies)

            # Update the final results
            test_results += [(f'{strategy.name}_{pair_time_frame_str}', result) for strategy, result in
                             zip(all_strategies, results)]

    # Sort the results so that the most profitable results are first
    test_results.sort(key=lambda x: x[1].net_reward, reverse=True)

    # Print the results
    print('\n----------------------------------------------------------')
    print('FINAL TEST RESULTS (ordered from most profitable to least)')
    print('----------------------------------------------------------')

    for name, res in test_results:
        print(name)
        print(res)
        print()

    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')


if __name__ == "__main__":
    test_strategies()
