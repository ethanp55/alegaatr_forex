import pickle
import random
from runner.simulation_runner import SimulationRunner
from strategies.strategy import Strategy
from typing import Dict, List, Optional


# Class that holds both the possible values and the concrete value for a specific genome feature
class GeneticFeature:
    def __init__(self, possible_values: Optional[List[any]] = None) -> None:
        self.possible_values, self.concrete_value = possible_values, None
        self.mutate()

    def mutate(self) -> None:
        assert self.possible_values is not None

        self.concrete_value = random.choice(self.possible_values)


# Genome class.  Each genome contains a strategy and a list of genetic features (class is above) that corresponds to
# the strategy's features
class Genome:
    def __init__(self, currency_pair: str, time_frame: str, strategy: Strategy) -> None:
        self.currency_pair, self.time_frame, self.strategy = currency_pair, time_frame, strategy
        self.features = self._initialize_features()
        self.set_strategy_attributes()

    # Each specific genome class must define the genetic features for the contained strategy
    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        pass

    # The performance function for a genome.  In this project, this should remain the same (the net reward after
    # using the genome's strategy and feature values to run a market simulation)
    def performance(self) -> float:
        simulation_results = SimulationRunner.run_simulation(self.strategy, self.currency_pair, self.time_frame,
                                                             optimize=True)

        return simulation_results.net_reward

    # Randomly changes some of the genome's feature values.  Once the feature values are mutated, the corresponding
    # strategy values need to be updated as well
    def mutate(self, possible_n_mutations_ratio: float) -> None:
        n_mutatable_features = len(self.features)
        max_mutations = int(n_mutatable_features * possible_n_mutations_ratio)
        if max_mutations == 0:
            max_mutations = n_mutatable_features
        possible_n_mutations = list(range(max_mutations))
        n_mutations = random.choice(possible_n_mutations)
        feature_mutation_indices = random.sample(list(range(n_mutatable_features)), n_mutations)
        keys = list(self.features.keys())

        for i in feature_mutation_indices:
            attribute_name = keys[i]
            self.features[attribute_name].mutate()

        self.set_strategy_attributes()

    # Function to map each concrete feature value (the value selected from the feature's possible values) to the
    # genome's strategy
    def set_strategy_attributes(self) -> None:
        for attribute_name, genetic_feature in self.features.items():
            self.strategy.__setattr__(attribute_name, genetic_feature.concrete_value)

            if attribute_name == 'lookback' or attribute_name == 'n_in_a_row':
                self.strategy.__setattr__('starting_idx', genetic_feature.concrete_value)

    # Saves the genome's features (useful for saving the features of the best genome as the genetic algorithm is run)
    def save_features(self) -> None:
        pair_time_frame_str = f'{self.currency_pair}_{self.time_frame}'
        strategy_name = self.strategy.name
        concrete_features_dict = {key: self.features[key].concrete_value for key in self.features.keys()}

        with open(f'../genetics/best_genome_features/{strategy_name}_{pair_time_frame_str}_features.pickle', 'wb') as f:
            pickle.dump(concrete_features_dict, f)


# Population class that represents a list of genomes
class Population:
    def __init__(self, genomes: List[Genome]) -> None:
        self.genomes = genomes
        self.performances = [genome.performance() for genome in self.genomes]
