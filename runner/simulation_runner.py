from data.data_loader import DataLoader
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.market_simulator import MarketSimulator
from strategies.strategy import Strategy


class SimulationRunner(object):
    @staticmethod
    def run_simulation(strategy: Strategy, currency_pair: str, time_frame: str,
                       optimize: bool) -> MarketSimulationResults:
        market_data_raw = DataLoader.load_simulation_data(currency_pair, 'M5', optimize)
        strategy_data_raw = DataLoader.load_simulation_data(currency_pair, time_frame, optimize, '2015-2023')

        return MarketSimulator.run_simulation(strategy, market_data_raw, strategy_data_raw, currency_pair)


from strategies.arima import ArimaStrategy

currency_pair = 'Eur_Usd'
time_frame = 'M30'
pair_time_frame_str = f'{currency_pair}_{time_frame}'
# model_name = f'KNN_{pair_time_frame_str}'
# model_name = f'MLP_{pair_time_frame_str}'
# model_name = f'RandomForest_{pair_time_frame_str}'
# model_name = f'LSTM_{pair_time_frame_str}'
# model_name = f'CNN_{pair_time_frame_str}'
# model_name = f'LstmMixture_{pair_time_frame_str}'
# model_name = f'ArimaLSTM_{pair_time_frame_str}'
model_name = f'Arima_{pair_time_frame_str}'
results = SimulationRunner.run_simulation(ArimaStrategy(model_name, close_trade_incrementally=False),
                                          currency_pair,
                                          time_frame,
                                          True)
print(results)
