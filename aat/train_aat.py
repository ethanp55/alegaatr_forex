from multiprocessing import Pool
from functools import partial
from runner.simulation_runner import SimulationRunner
from strategies.bar_movement import BarMovement
from strategies.beep_boop import BeepBoop
from strategies.bollinger_bands import BollingerBands
from strategies.choc import Choc
from strategies.keltner_channels import KeltnerChannels
from strategies.ma_crossover import MACrossover
from strategies.macd import MACD
from strategies.macd_key_level import MACDKeyLevel
from strategies.macd_stochastic import MACDStochastic
from strategies.psar import PSAR
from strategies.rsi import RSI
from strategies.squeeze_pro import SqueezePro
from strategies.stochastic import Stochastic
from strategies.supertrend import Supertrend
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def train_aat() -> None:
    # The "regular" strategies (just use a set of predetermined rules)
    strategy_types = [BarMovement, BeepBoop, BollingerBands, Choc, KeltnerChannels, MACrossover, MACD, MACDKeyLevel,
                      MACDStochastic, PSAR, RSI, SqueezePro, Stochastic, Supertrend, PSAR, RSI, Stochastic, Supertrend,
                      BeepBoop]

    for currency_pair in CURRENCY_PAIRS:
        for time_frame in TIME_FRAMES:
            for year in YEARS[:-1]:
                strategies_to_train = []

                # Make sure the "best" versions of each strategy are used for training AAT
                for strategy_type in strategy_types:
                    strategy = strategy_type()

                    try:
                        strategy.load_best_parameters(currency_pair, time_frame, year)
                        strategies_to_train.append(strategy)

                    except:
                        pass  # Do nothing if there are no "best" parameters for the strategy

                # Unlikely to happen, but if there are no "best" parameters for any of the strategies (the list of
                # strategies is empty), then just continue to the next pair-time combination
                if len(strategies_to_train) == 0:
                    continue

                pool = Pool(processes=len(strategies_to_train))
                pool.map(partial(SimulationRunner.run_simulation, currency_pair=currency_pair, time_frame=time_frame,
                                 year=year, optimize=True, train_aat=True), strategies_to_train)


if __name__ == "__main__":
    train_aat()
