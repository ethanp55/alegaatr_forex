from functools import partial
from genetics.alegaatr_genome import AlegAATrGenome
from genetics.bar_movement_genome import BarMovementGenome
from genetics.beep_boop_genome import BeepBoopGenome
from genetics.bollinger_bands_genome import BollingerBandsGenome
from genetics.choc_genome import ChocGenome
from genetics.cnn_genome import CNNGenome
from genetics.ensemble_genome import EnsembleGenome
from genetics.eee_genome import EEEGenome
from genetics.exp3_genome import EXP3Genome
from genetics.genetic_algorithm import GeneticAlgorithm
from genetics.keltner_channels_genome import KeltnerChannelsGenome
from genetics.knn_genome import KNNGenome
from genetics.lstm_genome import LstmGenome
from genetics.lstm_mixture_genome import LstmMixtureGenome
from genetics.ma_crossover_genome import MACrossoverGenome
from genetics.macd_genome import MACDGenome
from genetics.macd_key_level_genome import MACDKeyLevelGenome
from genetics.macd_stochastic_genome import MACDStochasticGenome
from genetics.mlp_genome import MLPGenome
from genetics.psar_genome import PSARGenome
from genetics.random_forest_genome import RandomForestGenome
from genetics.rsi_genome import RSIGenome
from genetics.squeeze_pro_genome import SqueezeProGenome
from genetics.stochastic_genome import StochasticGenome
from genetics.supertrend_genome import SupertrendGenome
from genetics.ucb_genome import UCBGenome
from multiprocessing import Pool
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def optimize_genomes() -> None:
    # genome_types = [MACDGenome, SqueezeProGenome, BarMovementGenome, BollingerBandsGenome, ChocGenome,
    #                 KeltnerChannelsGenome, MACrossoverGenome, MACDKeyLevelGenome, MACDStochasticGenome, PSARGenome,
    #                 RSIGenome, StochasticGenome, SupertrendGenome, BeepBoopGenome, KNNGenome, MLPGenome,
    #                 RandomForestGenome, CNNGenome, LstmGenome, EnsembleGenome, AlegAATrGenome, UCBGenome, EXP3Genome,
    #                 EEEGenome, LstmMixtureGenome]
    # genome_types = [LstmGenome, LstmMixtureGenome, CNNGenome, RandomForestGenome, KNNGenome, MLPGenome]
    genome_types = [LstmMixtureGenome]

    for genome_type in genome_types:
        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[1:-1]:
                    GeneticAlgorithm.run(genome_type, currency_pair, time_frame, year, n_iterations=4,
                                         population_size=4)

    # for currency_pair in CURRENCY_PAIRS:
    #     for time_frame in TIME_FRAMES:
    #         for year in YEARS[1:-1]:
    #             # Creates a new process for each genome type
    #             pool = Pool(processes=len(genome_types))
    #             pool.map(
    #                 partial(GeneticAlgorithm.run, currency_pair=currency_pair, time_frame=time_frame, year=year,
    #                         population_size=4, n_iterations=10),
    #                 genome_types)


if __name__ == "__main__":
    optimize_genomes()
