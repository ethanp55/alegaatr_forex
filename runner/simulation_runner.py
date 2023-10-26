from data.data_loader import DataLoader
from experiments.metrics_tracker import MetricsTracker
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.market_simulator import MarketSimulator
import numpy as np
from strategies.strategy import Strategy
from typing import Optional
from utils.utils import VALIDATION_START_DATE


class SimulationRunner(object):
    @staticmethod
    def run_simulation(strategy: Strategy, currency_pair: str, time_frame: str, optimize: bool,
                       train_aat: bool = False,
                       metrics_tracker: Optional[MetricsTracker] = None) -> MarketSimulationResults:
        market_data_raw = DataLoader.load_simulation_data(currency_pair, 'M5', optimize)
        strategy_data_raw = DataLoader.load_simulation_data(currency_pair, time_frame, optimize, '2015-2023')
        validation_market_data_raw, validation_strategy_data_raw = None, None

        # Create validation data
        if optimize:
            validation_market_data_raw = market_data_raw.loc[market_data_raw['Date'] >= VALIDATION_START_DATE]
            validation_market_data_raw.reset_index(drop=True, inplace=True)
            market_data_raw = market_data_raw.loc[market_data_raw['Date'] < VALIDATION_START_DATE]
            market_data_raw.reset_index(drop=True, inplace=True)

            validation_strategy_data_raw = strategy_data_raw.loc[strategy_data_raw['Date'] >= VALIDATION_START_DATE]
            validation_strategy_data_raw.reset_index(drop=True, inplace=True)
            strategy_data_raw = strategy_data_raw.loc[strategy_data_raw['Date'] < VALIDATION_START_DATE]
            strategy_data_raw.reset_index(drop=True, inplace=True)

        # If we're running on test data (i.e. we're not optimizing anything), load in the strategy's "best" parameters
        try:
            if not optimize:
                strategy.load_best_parameters(currency_pair, time_frame)

        except:
            # Case where there are no "best" parameters because they did not yield a positive net reward
            return MarketSimulationResults(-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                                           -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)

        results = MarketSimulator.run_simulation(strategy, market_data_raw, strategy_data_raw, currency_pair,
                                                 time_frame,
                                                 train_aat=train_aat, metrics_tracker=metrics_tracker)

        # Run another simulation, but this time on the validation data - add the validation net reward to the main
        # results.  Note that if we're training AAT, we don't want to run a validation simulation (don't want AAT to
        # have training instances from the validation set).
        if optimize and not train_aat:
            assert validation_market_data_raw is not None

            validation_results = MarketSimulator.run_simulation(strategy, validation_market_data_raw,
                                                                validation_strategy_data_raw,
                                                                currency_pair, time_frame)
            results.validation_net_reward = validation_results.net_reward

        return results
