from pandas import DataFrame
from market_proxy.trade import Trade
from typing import Callable, Optional


# Abstract strategy class that each specific strategy must implement
class Strategy:
    def __init__(self, starting_idx: int, data_format_function: Callable[[DataFrame], DataFrame],
                 percent_to_risk: float = 0.02) -> None:
        self.starting_idx = starting_idx
        self.data_format_function = data_format_function
        self.percent_to_risk = percent_to_risk

    # Each strategy has unique rules to determine if a trade should be placed
    def place_trade(self, curr_idx: int, market_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        pass

    # A strategy might have rules to move the stop loss if the market moves in the trade's favor
    def move_stop_loss(self, curr_idx: int, market_data: DataFrame, trade: Trade) -> Trade:
        return trade
