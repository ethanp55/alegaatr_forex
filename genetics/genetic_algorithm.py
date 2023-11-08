from copy import deepcopy
from genetics.genome import Genome, Population
import random
from typing import List, Tuple


# Class that contains the pieces of the genetic algorithm we use
class GeneticAlgorithm(object):
    # Runs the genetic algorithm
    @staticmethod
    def run(genome_type: type, currency_pair: str, time_frame: str, n_iterations: int = 5,
            population_size: int = 6, possible_n_mutations_ratio: float = 0.25) -> None:
        # Placeholders for the population and list of new genomes (used to updated the population)
        population, new_genomes = None, []

        # Iterate through the number of iterations for the genetic algorithm
        for i in range(n_iterations):
            print(f'Generation {i + 1} / {n_iterations}')

            # Initialize or update the population - whenever a new population is created, the performances are
            # automatically calculated
            population = GeneticAlgorithm._initialize_population(genome_type, currency_pair, time_frame,
                                                                 population_size) if population is None else Population(
                new_genomes)

            assert len(population.genomes) == population_size

            # Sort the genomes by performance, with the highest-performing genomes at the beginning (reverse=True)
            sorted_performances_with_indices = sorted(zip(enumerate(population.performances)), key=lambda x: x[0][-1],
                                                      reverse=True)

            # Grab the 2 best genomes and use them in the next population
            best_idx, best_performance = sorted_performances_with_indices[0][0]
            second_best_idx, second_best_performance = sorted_performances_with_indices[1][0]
            _, worst_performance = sorted_performances_with_indices[-1][0]
            best_genome, second_best_genome = population.genomes[best_idx], population.genomes[second_best_idx]

            strategy_name = best_genome.strategy.name

            print(f'{strategy_name} on {currency_pair} {time_frame}: best performance = {best_performance}, '
                  f'second-best performance = {second_best_performance}, worst performance = {worst_performance}')

            new_genomes = [best_genome, second_best_genome]

            # Save the features of the best genome
            if best_performance > 0:
                print(f'Updating {strategy_name} features that yielded {best_performance} on {currency_pair} '
                      f'{time_frame}')
                best_genome.save_features()

            # Generate the remainder of the new population
            for j in range(int(population_size / 2) - 1):
                # Randomly select 2 "parents"
                parents = GeneticAlgorithm._selection(population)

                # Create 2 "offspring" from the parents by performing a single point crossover
                offspring_a, offspring_b = GeneticAlgorithm._single_point_crossover(parents[0], parents[1])

                # Mutate the offspring
                offspring_a.mutate(possible_n_mutations_ratio)
                offspring_b.mutate(possible_n_mutations_ratio)

                # Add the 2 new genomes to the new population
                new_genomes += [offspring_a, offspring_b]

    # Initializes the genome population
    @staticmethod
    def _initialize_population(genome_type: type, currency_pair: str, time_frame: str,
                               population_size: int) -> Population:
        assert issubclass(genome_type, Genome)
        genomes = [genome_type(currency_pair, time_frame) for _ in range(population_size)]

        return Population(genomes)

    # Randomly selects 2 genomes from the population, where the random selection is weighted based on genome
    # performance.  The 2 genomes can be viewed as "parents"
    @staticmethod
    def _selection(population: Population) -> List[Genome]:
        performances = population.performances
        denominator = max(performances) - min(performances)

        if denominator != 0:
            weights = [(float(pop) - min(performances)) / denominator for pop in performances]
            weights = [weight / sum(weights) for weight in weights]

        else:
            weights = [1 / len(performances) for _ in performances]

        random_selection = random.choices(population.genomes, weights=weights, k=2)

        return [deepcopy(genome) for genome in random_selection]

    # Performs a single point crossover, where 2 genomes swap part of their features
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
