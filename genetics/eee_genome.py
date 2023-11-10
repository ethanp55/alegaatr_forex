from genetics.genome import GeneticFeature, Genome
from strategies.eee import EEE
from typing import Dict


class EEEGenome(Genome):
    def __init__(self, currency_pair: str, time_frame: str, year: int) -> None:
        super().__init__(currency_pair, time_frame, year, EEE(genetic=True))
        self.strategy.load_best_parameters(currency_pair, time_frame, year - 1)

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        min_num_predictions_feature = GeneticFeature([1, 2, 3, 5, 7])
        invert_feature = GeneticFeature([True, False])
        explore_prob_feature = GeneticFeature([0.05, 0.1, 0.15])

        feature_dictionary = {'min_num_predictions': min_num_predictions_feature,
                              'invert': invert_feature,
                              'explore_prob': explore_prob_feature
                              }

        return feature_dictionary
