from functools import partial
from genetics.genetic_algorithm import GeneticAlgorithm
from genetics.bar_movement_genome import BarMovementGenome
from genetics.beep_boop_genome import BeepBoopGenome
from genetics.bollinger_bands_genome import BollingerBandsGenome
from genetics.choc_genome import ChocGenome
from genetics.cnn_genome import CNNGenome
from genetics.ensemble_genome import EnsembleGenome
from genetics.keltner_channels_genome import KeltnerChannelsGenome
from genetics.knn_genome import KNNGenome
from genetics.lstm_genome import LstmGenome
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
from multiprocessing import Pool
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES


def optimize_genomes() -> None:
    # genome_types = [MACDGenome, SqueezeProGenome, BarMovementGenome, BollingerBandsGenome, ChocGenome,
    #                 KeltnerChannelsGenome, MACrossoverGenome, MACDKeyLevelGenome, MACDStochasticGenome, PSARGenome,
    #                 RSIGenome, StochasticGenome, SupertrendGenome, BeepBoopGenome, KNNGenome, MLPGenome,
    #                 RandomForestGenome]
    genome_types = [CNNGenome]
    # genome_types = [LstmGenome]
    # genome_types = [EnsembleGenome]

    for currency_pair in CURRENCY_PAIRS:
        for time_frame in TIME_FRAMES:
            # Creates a new process for each genome type
            pool = Pool(processes=len(genome_types))
            pool.map(partial(GeneticAlgorithm.run, currency_pair=currency_pair, time_frame=time_frame), genome_types)


if __name__ == "__main__":
    optimize_genomes()
