def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
from collections import deque
from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import Trade, TradeType
from models.cnn import CNN
import numpy as np
from pandas import DataFrame
from strategies.strategy import Strategy
from typing import Callable, Optional
from utils.technical_indicators import TechnicalIndicators


class CNNStrategy(Strategy):
    def __init__(self, model_name: str, starting_idx: int = 1,
                 data_format_function: Callable[
                     [DataFrame], DataFrame] = TechnicalIndicators.format_data_for_ml_model_for_simulation,
                 percent_to_risk: float = 0.02, ma_key: Optional[str] = 'smma200',
                 invert: bool = False, use_tsl: bool = False, pips_to_risk: Optional[int] = 50,
                 pips_to_risk_atr_multiplier: float = 2.0, risk_reward_ratio: Optional[float] = 1.5,
                 error_multiplier: float = 0.0,
                 close_trade_incrementally: bool = False) -> None:
        super().__init__(starting_idx, data_format_function, percent_to_risk, 'CNN')
        self.ma_key, self.invert, self.use_tsl, self.pips_to_risk, self.pips_to_risk_atr_multiplier, \
        self.risk_reward_ratio, self.error_multiplier, self.close_trade_incrementally = ma_key, invert, use_tsl, pips_to_risk, \
                                                                                        pips_to_risk_atr_multiplier, risk_reward_ratio, error_multiplier, close_trade_incrementally
        self.model = CNN(model_name)
        self.starting_idx = self.model.lookback
        self.model.load_model()
        self.ask_pips_up_errors, self.ask_pips_down_errors, self.bid_pips_up_errors, self.bid_pips_down_errors = deque(
            maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        # Grab the needed strategies values
        lower_atr_band1, upper_atr_band1, mid_close1 = strategy_data.loc[
            strategy_data.index[curr_idx - 1], ['lower_atr_band', 'upper_atr_band', 'Mid_Close']]
        ma = strategy_data.loc[strategy_data.index[curr_idx - 1], self.ma_key] if self.ma_key is not None else None
        pips_multiplier = 100 if 'Jpy' in currency_pair else 10000

        x = strategy_data.loc[
            strategy_data.index[curr_idx - self.model.lookback:curr_idx], ['bid_pips_down', 'bid_pips_up',
                                                                           'ask_pips_down', 'ask_pips_up', 'rsi',
                                                                           'rsi_sma', 'adx', 'chop', 'vo', 'qqe_up',
                                                                           'qqe_down', 'qqe_val',
                                                                           'rsi_up', 'adx_large', 'chop_small',
                                                                           'vo_positive', 'squeeze_on',
                                                                           'macd', 'macdsignal']]
        x['bid_pips_down'] *= pips_multiplier
        x['bid_pips_up'] *= pips_multiplier
        x['ask_pips_down'] *= pips_multiplier
        x['ask_pips_up'] *= pips_multiplier
        bid_pips_down_pred, bid_pips_up_pred, ask_pips_down_pred, ask_pips_up_pred, = self.model.predict(
            np.array(x))

        ask_pips_up_error_avg = np.array(self.ask_pips_up_errors).mean() if len(self.ask_pips_up_errors) > 0 else 0
        ask_pips_down_error_avg = np.array(self.ask_pips_down_errors).mean() if len(
            self.ask_pips_down_errors) > 0 else 0
        bid_pips_up_error_avg = np.array(self.bid_pips_up_errors).mean() if len(self.bid_pips_up_errors) > 0 else 0
        bid_pips_down_error_avg = np.array(self.bid_pips_down_errors).mean() if len(
            self.bid_pips_down_errors) > 0 else 0

        ask_pips_up_pred = ask_pips_up_pred + (ask_pips_up_error_avg * self.error_multiplier)
        ask_pips_down_pred = ask_pips_down_pred + (ask_pips_down_error_avg * self.error_multiplier)
        bid_pips_up_pred = bid_pips_up_pred + (bid_pips_up_error_avg * self.error_multiplier)
        bid_pips_down_pred = bid_pips_down_pred + (bid_pips_down_error_avg * self.error_multiplier)

        curr_date, curr_ao, curr_bo, curr_mo, curr_bh, curr_al = strategy_data.loc[
            strategy_data.index[curr_idx], ['Date', 'Ask_Open', 'Bid_Open', 'Mid_Open', 'Bid_High', 'Ask_Low']]
        divider = 100 if 'Jpy' in currency_pair else 10000
        pips_to_risk = self.pips_to_risk / divider if self.pips_to_risk is not None else None
        buy_pips_to_risk = pips_to_risk if pips_to_risk is not None else (curr_ao - lower_atr_band1) * \
                                                                         self.pips_to_risk_atr_multiplier
        sell_pips_to_risk = pips_to_risk if pips_to_risk is not None else (upper_atr_band1 - curr_bo) * \
                                                                          self.pips_to_risk_atr_multiplier

        # Determine if there is a buy or sell signal
        buy_signal = max([ask_pips_up_pred, ask_pips_down_pred, bid_pips_up_pred,
                          bid_pips_down_pred]) == bid_pips_up_pred and (
                         mid_close1 > ma if ma is not None else True)
        sell_signal = max([ask_pips_up_pred, ask_pips_down_pred, bid_pips_up_pred,
                           bid_pips_down_pred]) == ask_pips_down_pred and (
                          mid_close1 < ma if ma is not None else True)

        if self.invert:
            buy_signal, sell_signal = sell_signal, buy_signal

        # Adjust prediction errors
        ask_open, ask_high, ask_low, bid_open, bid_high, bid_low = strategy_data.loc[
            strategy_data.index[curr_idx], ['Ask_Open', 'Ask_High', 'Ask_Low', 'Bid_Open', 'Bid_High', 'Bid_Low']]
        ask_pips_up_true, ask_pips_down_true, bid_pips_up_true, bid_pips_down_true = abs(
            ask_high - ask_open) * pips_multiplier, abs(ask_open - ask_low) * pips_multiplier, abs(
            bid_high - bid_open) * pips_multiplier, abs(bid_open - bid_low) * pips_multiplier

        self.ask_pips_up_errors.append(abs(ask_pips_up_true - ask_pips_up_pred))
        self.ask_pips_down_errors.append(abs(ask_pips_down_true - ask_pips_down_pred))
        self.bid_pips_up_errors.append(abs(bid_pips_up_true - bid_pips_up_pred))
        self.bid_pips_down_errors.append(abs(bid_pips_down_true - bid_pips_down_pred))

        # If there is a signal, place a trade (assuming the spread is small enough)
        if buy_signal or sell_signal:
            spread = abs(curr_ao - curr_bo)
            sl_pips = buy_pips_to_risk if buy_signal else sell_pips_to_risk

            if buy_signal:
                open_price = curr_ao
                stop_loss = open_price - sl_pips

                if stop_loss < open_price and spread <= sl_pips * 0.1:
                    trade_type = TradeType.BUY
                    amount_to_risk = account_balance * self.percent_to_risk
                    n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                             currency_pair, amount_to_risk)
                    stop_gain = None if self.risk_reward_ratio is None else open_price + (
                            sl_pips * self.risk_reward_ratio)

                    return Trade(trade_type, open_price, stop_loss, stop_gain, n_units, sl_pips, curr_date,
                                 currency_pair)

            elif sell_signal:
                open_price = curr_bo
                stop_loss = open_price + sl_pips

                if stop_loss > open_price and spread <= sl_pips * 0.1:
                    trade_type = TradeType.SELL
                    amount_to_risk = account_balance * self.percent_to_risk
                    n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                             currency_pair, amount_to_risk)
                    stop_gain = None if self.risk_reward_ratio is None else open_price - (
                            sl_pips * self.risk_reward_ratio)

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
