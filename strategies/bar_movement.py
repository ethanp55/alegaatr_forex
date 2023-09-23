from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import Trade, TradeType
from pandas import DataFrame
from strategies.strategy import Strategy
from typing import Callable, Optional
from utils.technical_indicators import TechnicalIndicators


class BarMovement(Strategy):
    def __init__(self, starting_idx: int = 2,
                 data_format_function: Callable[
                     [DataFrame], DataFrame] = TechnicalIndicators.format_data_for_bar_movement,
                 percent_to_risk: float = 0.02, ma_key: Optional[str] = 'smma200',
                 invert: bool = False, use_tsl: bool = False, pips_to_risk: Optional[int] = 50,
                 pips_to_risk_atr_multiplier: float = 2.0, risk_reward_ratio: Optional[float] = 1.5,
                 close_to_level_atr_multiplier: float = 1.0, pip_movement_atr_multiplier: float = 2.0,
                 lookback: int = 12, close_trade_incrementally: bool = False) -> None:
        super().__init__(starting_idx, data_format_function, percent_to_risk, 'BarMovement')
        self.ma_key, self.invert, self.use_tsl, self.pips_to_risk, self.pips_to_risk_atr_multiplier, \
        self.risk_reward_ratio, self.close_to_level_atr_multiplier, self.pip_movement_atr_multiplier, self.lookback, self.close_trade_incrementally = ma_key, invert, use_tsl, pips_to_risk, \
                                                                                                                                                      pips_to_risk_atr_multiplier, risk_reward_ratio, close_to_level_atr_multiplier, pip_movement_atr_multiplier, lookback, close_trade_incrementally
        self.starting_idx = self.lookback

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        # Grab the needed strategies values
        mid_opens = list(strategy_data.loc[strategy_data.index[curr_idx - self.lookback:curr_idx], 'Mid_Open'])
        mid_closes = list(strategy_data.loc[strategy_data.index[curr_idx - self.lookback:curr_idx], 'Mid_Close'])
        atrs = list(strategy_data.loc[strategy_data.index[curr_idx - self.lookback:curr_idx], 'atr'])
        avg_atr = sum(atrs) / len(atrs)

        atr1, lower_atr_band1, upper_atr_band1, mid_close1 = strategy_data.loc[
            strategy_data.index[curr_idx - 1], ['atr', 'lower_atr_band', 'upper_atr_band', 'Mid_Close']]
        ma = strategy_data.loc[strategy_data.index[curr_idx - 1], self.ma_key] if self.ma_key is not None else None
        pip_movement = avg_atr * self.pip_movement_atr_multiplier
        pips_level_rounding = 0 if 'Jpy' in currency_pair else 2

        # Determine if there is a buy or sell signal
        def _check_bars(buy: bool) -> bool:
            pips_moved = 0

            for j in range(len(mid_opens) - 1, -1, -1):
                if (buy and mid_opens[j] < mid_closes[j]) or (not buy and mid_opens[j] > mid_closes[j]):
                    pips_moved += abs(mid_opens[j] - mid_closes[j])

                    if pips_moved >= pip_movement:
                        return True

                else:
                    return False

            return False

        nearest_level = round(mid_close1, pips_level_rounding)
        close_to_nearest_level = abs(mid_close1 - nearest_level) < atr1 * self.close_to_level_atr_multiplier

        buy_signal = close_to_nearest_level and (mid_close1 > ma if ma is not None else True) and _check_bars(buy=True)
        sell_signal = close_to_nearest_level and (mid_close1 < ma if ma is not None else True) and _check_bars(
            buy=False)

        if self.invert:
            buy_signal, sell_signal = sell_signal, buy_signal

        # If there is a signal, place a trade (assuming the spread is small enough)
        if buy_signal or sell_signal:
            curr_date, curr_ao, curr_bo, curr_mo, curr_bh, curr_al = strategy_data.loc[
                strategy_data.index[curr_idx], ['Date', 'Ask_Open', 'Bid_Open', 'Mid_Open', 'Bid_High', 'Ask_Low']]
            spread = abs(curr_ao - curr_bo)
            divider = 100 if 'Jpy' in currency_pair else 10000
            pips_to_risk = self.pips_to_risk / divider if self.pips_to_risk is not None else None

            if buy_signal:
                open_price = curr_ao
                sl_pips = pips_to_risk if pips_to_risk is not None else (open_price - lower_atr_band1) * \
                                                                        self.pips_to_risk_atr_multiplier
                stop_loss = open_price - sl_pips

                if stop_loss < open_price and spread <= sl_pips * 0.1 and curr_al <= open_price:
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
                sl_pips = pips_to_risk if pips_to_risk is not None else (upper_atr_band1 - open_price) * \
                                                                        self.pips_to_risk_atr_multiplier
                stop_loss = open_price + sl_pips

                if stop_loss > open_price and spread <= sl_pips * 0.1 and curr_bh >= open_price:
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
