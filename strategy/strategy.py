from copy import copy
from pandas import DataFrame
from market_proxy.trade import Trade, TradeType
from typing import Callable, Optional


# Abstract strategy class that each specific strategy must implement
class Strategy:
    def __init__(self, starting_idx: int, data_format_function: Callable[[DataFrame], DataFrame],
                 percent_to_risk: float) -> None:
        self.starting_idx = starting_idx
        self.data_format_function = data_format_function
        self.percent_to_risk = percent_to_risk

    # Each strategy has unique rules to determine if a trade should be placed
    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        pass

    # A strategy might have rules to move the stop loss if the market moves in the trade's favor
    def move_stop_loss(self, curr_idx: int, market_data: DataFrame, trade: Trade) -> Trade:
        return trade


# A function that simulates a basic trailing stop loss
def basic_tsl(curr_idx: int, market_data: DataFrame, trade: Trade) -> Trade:
    trade_copy = copy(trade)
    curr_bid_high, curr_ask_low = market_data.loc[market_data.index[curr_idx], ['Bid_High', 'Ask_Low']]

    # Move the stop loss on a buy if the market is in our favor
    if trade_copy.trade_type == TradeType.BUY and curr_bid_high - trade_copy.pips_risked > trade_copy.stop_loss:
        trade_copy.stop_loss = curr_bid_high - trade_copy.pips_risked

    elif trade_copy.trade_type == TradeType.SELL and curr_ask_low + trade_copy.pips_risked < trade_copy.stop_loss:
        trade_copy.stop_loss = curr_ask_low + trade_copy.pips_risked

    return trade_copy
