from genetics.genome import GeneticFeature, Genome
from strategies.ucb import UCB
from typing import Dict


class UCBGenome(Genome):
    def __init__(self, currency_pair: str, time_frame: str, year: int) -> None:
        super().__init__(currency_pair, time_frame, year, UCB(genetic=True))
        self.strategy.load_best_parameters(currency_pair, time_frame, year - 1)

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        delta_feature = GeneticFeature([0.9, 0.95, 0.99])

        feature_dictionary = {'delta': delta_feature}

        return feature_dictionary
