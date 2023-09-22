from copy import deepcopy
from genetics.genome import Genome, Population
import random
from typing import List, Tuple


class GeneticAlgorithm(object):
    @staticmethod
    def run(genome_type: type, currency_pair: str, time_frame: str, n_iterations: int = 50,
            population_size: int = 25, possible_n_mutations_ratio: float = 0.25) -> None:
        population = GeneticAlgorithm._initialize_population(genome_type, currency_pair, time_frame, population_size)

        for i in range(n_iterations):
            print(f'Generation {i + 1} / {n_iterations}')

            assert len(population.genomes) == population_size

            sorted_performances_with_indices = sorted(zip(enumerate(population.performances)), key=lambda x: x[0][-1],
                                                      reverse=True)
            best_idx, best_performance = sorted_performances_with_indices[0][0]
            second_best_idx, second_best_performance = sorted_performances_with_indices[1][0]
            _, worst_performance = sorted_performances_with_indices[-1][0]
            best_genome, second_best_genome = population.genomes[best_idx], population.genomes[second_best_idx]

            print(
                f'Best performance = {best_performance}, second-best performance = {second_best_performance}, worst performance = {worst_performance}')

            new_genomes = [best_genome, second_best_genome]

            for j in range(int(population_size / 2) - 1):
                parents = GeneticAlgorithm._selection(population)
                offspring_a, offspring_b = GeneticAlgorithm._single_point_crossover(parents[0], parents[1])
                offspring_a.mutate(possible_n_mutations_ratio)
                offspring_b.mutate(possible_n_mutations_ratio)
                new_genomes += [offspring_a, offspring_b]

            population = Population(new_genomes)

    @staticmethod
    def _initialize_population(genome_type: type, currency_pair: str, time_frame: str,
                               population_size: int) -> Population:
        assert issubclass(genome_type, Genome)
        genomes = [genome_type(currency_pair, time_frame) for _ in range(population_size)]

        return Population(genomes)

    @staticmethod
    def _selection(population: Population, k: int = 2) -> List[Genome]:
        random_selection = random.choices(population.genomes, weights=population.performances, k=k)
        return [deepcopy(genome) for genome in random_selection]

    @staticmethod
    def _single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        assert type(a) == type(b)

        new_a, new_b = a, b

        feature_length = len(new_a.features)
        idx = random.randint(1, feature_length - 1)

        assert (new_a.features.keys() == new_b.features.keys())

        keys = list(new_a.features.keys())

        for key in keys[idx:]:
            new_a.features[key].concrete_value, new_b.features[key].concrete_value = \
                new_b.features[key].concrete_value, new_a.features[key].concrete_value

        new_a.set_strategy_attributes()
        new_b.set_strategy_attributes()

        return new_a, new_b
