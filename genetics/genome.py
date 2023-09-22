import random
from runner.simulation_runner import SimulationRunner
from strategies.strategy import Strategy
from typing import Dict, List, Optional


class GeneticFeature:
    def __init__(self, possible_values: Optional[List[any]] = None) -> None:
        self.possible_values, self.concrete_value = possible_values, None
        self.mutate()

    def mutate(self) -> None:
        assert self.possible_values is not None

        self.concrete_value = random.choice(self.possible_values)


class Genome:
    def __init__(self, currency_pair: str, time_frame: str, strategy: Strategy) -> None:
        self.currency_pair, self.time_frame, self.strategy = currency_pair, time_frame, strategy
        self.features = self._initialize_features()
        self.set_strategy_attributes()

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        pass

    def performance(self) -> float:
        simulation_results = SimulationRunner.run_simulation(self.strategy, self.currency_pair, self.time_frame,
                                                             optimize=True)

        return simulation_results.net_reward

    def mutate(self, possible_n_mutations_ratio: float) -> None:
        n_mutatable_features = len(self.features)
        possible_n_mutations = list(range(int(n_mutatable_features * possible_n_mutations_ratio)))
        n_mutations = random.choice(possible_n_mutations)
        feature_mutation_indices = random.sample(list(range(n_mutatable_features)), n_mutations)
        keys = list(self.features.keys())

        for i in feature_mutation_indices:
            attribute_name = keys[i]
            self.features[attribute_name].mutate()

        self.set_strategy_attributes()

    def set_strategy_attributes(self) -> None:
        for attribute_name, genetic_feature in self.features.items():
            self.strategy.__setattr__(attribute_name, genetic_feature.concrete_value)


class Population:
    def __init__(self, genomes: List[Genome]) -> None:
        self.genomes = genomes
        self.performances = [genome.performance() for genome in self.genomes]
