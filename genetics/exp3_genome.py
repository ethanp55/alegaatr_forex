from genetics.genome import GeneticFeature, Genome
from strategies.exp3 import EXP3
from typing import Dict


class EXP3Genome(Genome):
    def __init__(self, currency_pair: str, time_frame: str, year: int) -> None:
        super().__init__(currency_pair, time_frame, year, EXP3(genetic=True))
        self.strategy.load_best_parameters(currency_pair, time_frame, year - 1)

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        min_num_predictions_feature = GeneticFeature([1, 2, 3, 5, 7])
        invert_feature = GeneticFeature([True, False])
        gamma_feature = GeneticFeature([0.4, 0.5, 0.6])

        feature_dictionary = {'min_num_predictions': min_num_predictions_feature,
                              'invert': invert_feature,
                              'gamma': gamma_feature
                              }

        return feature_dictionary
