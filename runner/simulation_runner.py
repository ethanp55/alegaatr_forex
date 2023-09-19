from data.data_loader import DataLoader
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.market_simulator import MarketSimulator
from strategy.strategy import Strategy


class SimulationRunner(object):
    @staticmethod
    def run_simulation(strategy: Strategy, currency_pair: str, time_frame: str,
                       optimize: bool) -> MarketSimulationResults:
        market_data_raw = DataLoader.load_simulation_data(currency_pair, 'M5', optimize)
        strategy_data_raw = DataLoader.load_simulation_data(currency_pair, time_frame, optimize, '2015-2023')

        return MarketSimulator.run_simulation(strategy, market_data_raw, strategy_data_raw, currency_pair)


from strategy.bar_movement import BarMovement

results = SimulationRunner.run_simulation(BarMovement(close_trade_incrementally=True), 'Eur_Usd', 'M30', True)
print(results)
