from data.data_loader import DataLoader
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.market_simulator import MarketSimulator
import numpy as np
from strategies.strategy import Strategy


class SimulationRunner(object):
    @staticmethod
    def run_simulation(strategy: Strategy, currency_pair: str, time_frame: str, optimize: bool,
                       train_aat: bool = False) -> MarketSimulationResults:
        market_data_raw = DataLoader.load_simulation_data(currency_pair, 'M5', optimize)
        strategy_data_raw = DataLoader.load_simulation_data(currency_pair, time_frame, optimize, '2015-2023')

        # If we're running on test data (i.e. we're not optimizing anything), load in the strategy's "best" parameters
        try:
            if not optimize:
                strategy.load_best_parameters(currency_pair, time_frame)

        except:
            # Case where there are no "best" parameters because they did not yield a positive net reward
            return MarketSimulationResults(-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                                           -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)

        return MarketSimulator.run_simulation(strategy, market_data_raw, strategy_data_raw, currency_pair, time_frame,
                                              train_aat=train_aat)
