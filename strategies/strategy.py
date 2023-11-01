from copy import deepcopy
from numpy import floor
from pandas import DataFrame
import pickle
from market_proxy.trade import Trade, TradeType
from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulation_results import MarketSimulationResults
from typing import Optional


# Abstract strategies class that each specific strategies must implement
class Strategy:
    def __init__(self, starting_idx: int, percent_to_risk: float, name: str) -> None:
        self.starting_idx = starting_idx
        self.percent_to_risk = percent_to_risk
        self.name = name

    def print_parameters(self) -> None:
        for name, val in self.__dict__.items():
            print(f'{name}: {val}')

    # Loads in the best parameter values that were estimated by the genetic algorithm
    def load_best_parameters(self, currency_pair: str, time_frame: str) -> None:
        pair_time_frame_str = f'{currency_pair}_{time_frame}'
        best_params_dictionary = pickle.load(
            open(f'../genetics/best_genome_features/{self.name}_{pair_time_frame_str}_features.pickle', 'rb'))

        for attribute_name, val in best_params_dictionary.items():
            self.__setattr__(attribute_name, val)

            if attribute_name == 'lookback' or attribute_name == 'n_in_a_row':
                self.__setattr__('starting_idx', val)

    # Each strategies has unique rules to determine if a trade should be placed
    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        pass

    # A strategies might have rules to move the stop loss if the market moves in the trade's favor
    def move_stop_loss(self, curr_idx: int, market_data: DataFrame, trade: Trade) -> Trade:
        trade_copy = deepcopy(trade)
        curr_bid_high, curr_ask_low = market_data.loc[market_data.index[curr_idx], ['Bid_High', 'Ask_Low']]

        # Move the stop loss on a buy if the market is in our favor
        if trade_copy.trade_type == TradeType.BUY and curr_bid_high - trade_copy.pips_risked > trade_copy.stop_loss:
            trade_copy.stop_loss = curr_bid_high - trade_copy.pips_risked

        elif trade_copy.trade_type == TradeType.SELL and curr_ask_low + trade_copy.pips_risked < trade_copy.stop_loss:
            trade_copy.stop_loss = curr_ask_low + trade_copy.pips_risked

        return trade_copy

    # A strategies might want to close part of the trade and move the stop loss after the market has moved a certain
    # amount in the trade's favor
    def close_part_of_trade(self, curr_idx: int, market_data: DataFrame, trade: Trade,
                            simulation_results: MarketSimulationResults, currency_pair: str) -> Optional[Trade]:
        trade_copy = deepcopy(trade)
        # return trade_copy
        trade_type = trade_copy.trade_type
        curr_date, curr_bid_close, curr_ask_close = market_data.loc[
            market_data.index[curr_idx], ['Date', 'Bid_Close', 'Ask_Close']]
        pips_gained = (curr_bid_close - trade_copy.open_price) if trade_type == TradeType.BUY else (
                trade_copy.open_price - curr_ask_close)

        if pips_gained > 0:
            curr_profit_ratio = floor(pips_gained / trade_copy.pips_risked)
            new_stop_loss_ratio = curr_profit_ratio - 1.0
            new_stop_loss = (trade_copy.open_price + (
                    new_stop_loss_ratio * trade_copy.pips_risked)) if trade_type == TradeType.BUY else (
                    trade_copy.open_price - (new_stop_loss_ratio * trade_copy.pips_risked))

            if (trade_type == TradeType.BUY and new_stop_loss > trade_copy.stop_loss) or (
                    trade_type == TradeType.SELL and new_stop_loss < trade_copy.stop_loss):
                half_units = int(trade_copy.n_units / 2)

                if half_units == 0:
                    return None

                trade_copy.n_units = half_units
                trade_amount = pips_gained * trade_copy.n_units
                day_fees = MarketCalculations.calculate_day_fees(trade_copy, currency_pair, curr_date)
                simulation_results.update_results(trade_amount, day_fees)
                trade_copy.start_date = curr_date
                trade_copy.stop_loss = new_stop_loss

        return trade_copy
