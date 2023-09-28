from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import Trade, TradeType
from pandas import DataFrame
from strategies.strategy import Strategy
from typing import Callable, List, Optional
from utils.technical_indicators import TechnicalIndicators


class Ensemble(Strategy):
    def __init__(self, starting_idx: int = 1,
                 data_format_function: Callable[
                     [DataFrame], DataFrame] = TechnicalIndicators.format_for_all_possible_strategies,
                 percent_to_risk: float = 0.02, strategy_pool: List[Strategy] = [],
                 min_num_predictions: int = 3) -> None:
        super().__init__(starting_idx, data_format_function, percent_to_risk, 'Ensemble')
        self.strategy_pool, self.min_num_predictions = strategy_pool, min_num_predictions
        self.use_tsl, self.close_trade_incrementally = False, False
        self.min_idx = 0

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        for strategy in self.strategy_pool:
            self.min_idx = max(self.min_idx, strategy.starting_idx)

        # Safety check
        if len(self.strategy_pool) == 0 or curr_idx < self.min_idx:
            return None

        # Keep track of different metrics that the strategies "vote" on
        n_buys, n_sells = 0, 0
        n_buy_use_tsl, n_sell_use_tsl, n_buy_close_trade_incrementally, n_sell_close_trade_incrementally = 0, 0, 0, 0
        buy_sl_pips, sell_sl_pips = [], []
        buy_sg_pips, sell_sg_pips = [], []

        # Current data used for trade calculations
        curr_date, curr_ao, curr_bo, curr_mo, curr_bh, curr_al = strategy_data.loc[
            strategy_data.index[curr_idx], ['Date', 'Ask_Open', 'Bid_Open', 'Mid_Open', 'Bid_High', 'Ask_Low']]
        spread = abs(curr_ao - curr_bo)

        # Use each strategy in the pool to "vote"
        for strategy in self.strategy_pool:
            trade = strategy.place_trade(curr_idx, strategy_data, currency_pair, account_balance)

            if trade is not None:
                if trade.trade_type == TradeType.BUY:
                    n_buys += 1
                    n_buy_use_tsl += 1 if (hasattr(strategy, 'use_tsl') and strategy.use_tsl) else 0
                    n_buy_close_trade_incrementally += 1 if \
                        (hasattr(strategy, 'close_trade_incrementally') and strategy.close_trade_incrementally) else 0
                    buy_sl_pips.append(trade.pips_risked)

                    if trade.stop_gain is not None:
                        buy_sg_pips.append(trade.stop_gain - curr_ao)

                else:
                    n_sells += 1
                    n_sell_use_tsl += 1 if (hasattr(strategy, 'use_tsl') and strategy.use_tsl) else 0
                    n_sell_close_trade_incrementally += 1 if \
                        (hasattr(strategy, 'close_trade_incrementally') and strategy.close_trade_incrementally) else 0
                    sell_sl_pips.append(trade.pips_risked)

                    if trade.stop_gain is not None:
                        sell_sg_pips.append(curr_bo - trade.stop_gain)

        # If there are more buy votes than sell votes and the number of buy votes meets the minimum number of votes,
        # place a buy
        if n_buys > n_sells and n_buys >= self.min_num_predictions:
            open_price = curr_ao
            sl_pips = sum(buy_sl_pips) / len(buy_sl_pips)
            stop_loss = open_price - sl_pips

            if stop_loss < open_price and spread <= sl_pips * 0.1:
                trade_type = TradeType.BUY
                amount_to_risk = account_balance * self.percent_to_risk
                n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                         currency_pair, amount_to_risk)
                stop_gain = None if len(buy_sg_pips) < n_buys * 0.5 else open_price + (
                        sum(buy_sg_pips) / len(buy_sg_pips))

                self.use_tsl = n_buy_use_tsl >= n_buys * 0.5
                self.close_trade_incrementally = n_buy_close_trade_incrementally >= n_buys * 0.5

                return Trade(trade_type, open_price, stop_loss, stop_gain, n_units, sl_pips, curr_date,
                             currency_pair)

        # If there are more sell votes than buy votes and the number of sell votes meets the minimum number of votes,
        # place a sell
        elif n_sells > n_buys and n_sells >= self.min_num_predictions:
            open_price = curr_bo
            sl_pips = sum(sell_sl_pips) / len(sell_sl_pips)
            stop_loss = open_price + sl_pips

            if stop_loss > open_price and spread <= sl_pips * 0.1:
                trade_type = TradeType.SELL
                amount_to_risk = account_balance * self.percent_to_risk
                n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                         currency_pair, amount_to_risk)
                stop_gain = None if len(sell_sg_pips) < n_sells * 0.5 else open_price - (
                        sum(sell_sg_pips) / len(sell_sg_pips))

                self.use_tsl = n_sell_use_tsl >= n_sells * 0.5
                self.close_trade_incrementally = n_sell_close_trade_incrementally >= n_sells * 0.5

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
