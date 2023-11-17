from genetics.genome import GeneticFeature, Genome
from strategies.alegaatr import AlegAATr
from typing import Dict


class AlegAATrGenome(Genome):
    def __init__(self, currency_pair: str, time_frame: str) -> None:
        super().__init__(currency_pair, time_frame, AlegAATr(genetic=True))
        self.strategy.load_best_parameters(currency_pair, time_frame)

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        optimistic_start_feature = GeneticFeature([True, False])
        lmbda_feature = GeneticFeature([0.9, 0.95, 0.99, 0.995, 0.999])

        feature_dictionary = {'optimistic_start': optimistic_start_feature,
                              'lmbda': lmbda_feature}

        return feature_dictionary
