from genetics.genome import GeneticFeature, Genome
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
from typing import Dict


class EnsembleGenome(Genome):
    def __init__(self, currency_pair: str, time_frame: str) -> None:
        super().__init__(currency_pair, time_frame, Ensemble())

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        pair_time_frame_str = f'{self.currency_pair}_{self.time_frame}'
        cnn_model_name = f'CNN_{pair_time_frame_str}'
        knn_model_name = f'KNN_{pair_time_frame_str}'
        lstm_model_name = f'LSTM_{pair_time_frame_str}'
        mlp_model_name = f'MLP_{pair_time_frame_str}'
        rf_model_name = f'RandomForest_{pair_time_frame_str}'

        # all_strategies = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(), MACD(),
        #                   MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(), Supertrend(),
        #                   CNNStrategy(cnn_model_name), KNNStrategy(knn_model_name), LstmStrategy(lstm_model_name),
        #                   MLPStrategy(mlp_model_name), RandomForestStrategy(rf_model_name)]
        all_strategies = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(), MACD(),
                          MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(), Supertrend()]
        strategies = []

        for strategy in all_strategies:
            try:
                strategy.load_best_parameters(self.currency_pair, self.time_frame)
                strategies.append(strategy)

            except:
                # If the strategy doesn't have parameters for the current pair and time frame, skip it
                pass

        def _powerset(s):
            x = len(s)
            masks = [1 << i for i in range(x)]

            for i in range(1 << x):
                yield [ss for mask, ss in zip(masks, s) if i & mask]

        # Calculate all possible strategy combinations (the powerset, or set of all subsets)
        strategies_powerset = _powerset(strategies)

        # Make sure to only use strategy subsets that have at least 2 strategies in them (a single strategy is the
        # same as using that strategy by itself, which we already have code for)
        filtered_strategies_powerset = []

        for strategy_subset in strategies_powerset:
            if len(strategy_subset) >= 2:
                filtered_strategies_powerset.append(strategy_subset)

        strategy_pool_feature = GeneticFeature(filtered_strategies_powerset)
        min_num_predictions_feature = GeneticFeature([2, 3, 5, 7, 9])

        feature_dictionary = {'strategy_pool': strategy_pool_feature,
                              'min_num_predictions': min_num_predictions_feature}

        return feature_dictionary
