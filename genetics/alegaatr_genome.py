from genetics.genome import GeneticFeature, Genome
from strategies.alegaatr import AlegAATr
from typing import Dict


class AlegAATrGenome(Genome):
    def __init__(self, currency_pair: str, time_frame: str) -> None:
        super().__init__(currency_pair, time_frame, AlegAATr())
        self.strategy.load_best_parameters(currency_pair, time_frame)

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        min_num_predictions_feature = GeneticFeature([1, 2, 3, 5, 7, 9])
        use_single_selection_feature = GeneticFeature([True, False])
        min_n_neighbors_feature = GeneticFeature([5, 10, 15, 30, 60])
        lmbda_feature = GeneticFeature([0.95, 0.99, 0.999])

        feature_dictionary = {'min_num_predictions': min_num_predictions_feature,
                              # 'use_single_selection': use_single_selection_feature,
                              'min_n_neighbors': min_n_neighbors_feature,
                              'lmbda': lmbda_feature
                              }

        return feature_dictionary
