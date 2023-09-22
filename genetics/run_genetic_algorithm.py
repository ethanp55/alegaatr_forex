from genetics.genetic_algorithm import GeneticAlgorithm
from genetics.macd_genome import MACDGenome
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES


def optimize_genomes() -> None:
    genome_types = [MACDGenome]

    for currency_pair in CURRENCY_PAIRS:
        for time_frame in TIME_FRAMES:
            for genome_type in genome_types:
                print(f'Optimizing {str(genome_type)} on {currency_pair} {time_frame}')
                GeneticAlgorithm.run(genome_type, currency_pair, time_frame)


if __name__ == "__main__":
    optimize_genomes()
