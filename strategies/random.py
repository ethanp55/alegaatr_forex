from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import Trade, TradeType
import numpy as np
from pandas import DataFrame
import random
from strategies.strategy import Strategy
from typing import Optional


class Random(Strategy):
    def __init__(self, starting_idx: int = 0, percent_to_risk: float = 0.02,
                 close_trade_incrementally: bool = False) -> None:
        super().__init__(starting_idx, percent_to_risk, 'Random001')
        self.close_trade_incrementally = close_trade_incrementally
        self.use_tsl = False

    def load_best_parameters(self, currency_pair: str, time_frame: str, year: int) -> None:
        # This strategy does not have any tunable parameters
        return

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        selection = np.random.choice(['buy', 'sell', 'nothing'], p=[0.001, 0.001, 0.998])
        buy_signal, sell_signal = selection == 'buy', selection == 'sell'

        # If there is a signal, place a trade (assuming the spread is small enough)
        if buy_signal or sell_signal:
            curr_date, curr_ao, curr_bo, curr_mo, curr_bh, curr_al = strategy_data.loc[
                strategy_data.index[curr_idx], ['Date', 'Ask_Open', 'Bid_Open', 'Mid_Open', 'Bid_High', 'Ask_Low']]
            spread = abs(curr_ao - curr_bo)
            divider = 100 if 'Jpy' in currency_pair else 10000
            pips_to_risk = random.randint(10, 100) / divider

            if buy_signal:
                open_price = curr_ao
                sl_pips = pips_to_risk
                stop_loss = open_price - sl_pips
                # stop_loss = -np.inf

                if stop_loss < open_price and spread <= sl_pips * 0.1:
                    trade_type = TradeType.BUY
                    amount_to_risk = account_balance * self.percent_to_risk
                    n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                             currency_pair, amount_to_risk)
                    use_stop_gain = random.choice([True, False])
                    # pips_gained = None if not use_stop_gain else random.randint(10, 100) / divider
                    pips_gained = 50 / divider
                    stop_gain = None if pips_gained is None else open_price + pips_gained
                    # stop_gain = None

                    self.use_tsl = True if stop_gain is None else False

                    return Trade(trade_type, open_price, stop_loss, stop_gain, n_units, sl_pips, curr_date,
                                 currency_pair)

            elif sell_signal:
                open_price = curr_bo
                sl_pips = pips_to_risk
                stop_loss = open_price + sl_pips
                # stop_loss = np.inf

                if stop_loss > open_price and spread <= sl_pips * 0.1:
                    trade_type = TradeType.SELL
                    amount_to_risk = account_balance * self.percent_to_risk
                    n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                             currency_pair, amount_to_risk)
                    use_stop_gain = random.choice([True, False])
                    # pips_gained = None if not use_stop_gain else random.randint(10, 100) / divider
                    pips_gained = 50 / divider
                    stop_gain = None if pips_gained is None else open_price - pips_gained
                    # stop_gain = None

                    self.use_tsl = True if stop_gain is None else False

                    return Trade(trade_type, open_price, stop_loss, stop_gain, n_units, sl_pips, curr_date,
                                 currency_pair)

        return None

    def move_stop_loss(self, curr_idx: int, market_data: DataFrame, trade: Trade) -> Trade:
        if self.use_tsl:
            return super().move_stop_loss(curr_idx, market_data, trade)

        else:
            return trade

    def close_part_of_trade(self, curr_idx: int, market_data: DataFrame, trade: Trade,
                            simulation_results: MarketSimulationResults, currency_pair: str) -> Optional[Trade]:
        if self.close_trade_incrementally:
            return super().close_part_of_trade(curr_idx, market_data, trade, simulation_results, currency_pair)

        else:
            return trade
