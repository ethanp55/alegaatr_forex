from genetics.genome import GeneticFeature, Genome
from strategies.alegaatr import AlegAATr
from typing import Dict


class AlegAATrGenome(Genome):
    def __init__(self, currency_pair: str, time_frame: str, year: int) -> None:
        super().__init__(currency_pair, time_frame, year, AlegAATr())
        self.strategy.load_best_parameters(currency_pair, time_frame, year)

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        min_num_predictions_feature = GeneticFeature([1, 2, 3, 5, 7, 9])
        use_single_selection_feature = GeneticFeature([True, False])
        min_neighbors_feature = GeneticFeature([5, 10, 15, 30, 45, 60])
        max_neighbors_feature = GeneticFeature([60, 90, 120, 150, 180, 300, 100000])

        feature_dictionary = {'min_num_predictions': min_num_predictions_feature,
                              'use_single_selection': use_single_selection_feature,
                              'min_neighbors': min_neighbors_feature,
                              'max_neighbors': max_neighbors_feature}

        return feature_dictionary
