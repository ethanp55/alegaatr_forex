from copy import deepcopy
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import Trade, TradeType
import numpy as np
from pandas import DataFrame
from strategies.bar_movement import BarMovement
from strategies.beep_boop import BeepBoop
from strategies.bollinger_bands import BollingerBands
from strategies.choc import Choc
from strategies.keltner_channels import KeltnerChannels
from strategies.ma_crossover import MACrossover
from strategies.macd import MACD
from strategies.macd_key_level import MACDKeyLevel
from strategies.macd_stochastic import MACDStochastic
from strategies.psar import PSAR
from strategies.rsi import RSI
from strategies.squeeze_pro import SqueezePro
from strategies.stochastic import Stochastic
from strategies.strategy import Strategy
from strategies.supertrend import Supertrend
from typing import Optional


class UCB(Strategy):
    def __init__(self, starting_idx: int = 1, percent_to_risk: float = 0.02, delta: float = 0.99,
                 genetic: bool = False, min_num_predictions: int = 1, invert: bool = False) -> None:
        super().__init__(starting_idx, percent_to_risk, 'UCB')
        strategy_pool = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(),
                         MACD(), MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(),
                         Supertrend()]
        self.experts, self.empirical_rewards, self.n_samples = {}, {}, {}
        self.use_tsl, self.close_trade_incrementally = False, False
        self.min_idx = 0

        for expert in strategy_pool:
            self.experts[expert.name] = expert
            self.empirical_rewards[expert.name] = 0
            self.n_samples[expert.name] = 0

        self.delta, self.genetic, self.min_num_predictions, self.invert = delta, genetic, min_num_predictions, invert
        self.expert_name = None

    def load_best_parameters(self, currency_pair: str, time_frame: str, year: int) -> None:
        if not self.genetic:
            super().load_best_parameters(currency_pair, time_frame, year)

        for generator in self.experts.values():
            try:
                # Make sure the generator is using its "best" parameters for the given currency pair and time frame
                generator.load_best_parameters(currency_pair, time_frame, year)

            except:
                continue

    def trade_finished(self, net_profit: float) -> None:
        self.empirical_rewards[self.expert_name] += net_profit

    def _invert(self, trade: Trade, curr_ao: float, curr_bo: float) -> Trade:
        if self.invert:
            trade_copy = deepcopy(trade)

            if trade.trade_type == TradeType.BUY:
                trade_copy.trade_type = TradeType.SELL
                trade_copy.open_price = curr_bo
                trade_copy.stop_loss = curr_bo + trade.pips_risked

                if trade.stop_gain is not None:
                    stop_gain_pips = abs(trade.stop_gain - trade.open_price)
                    trade_copy.stop_gain = curr_bo - stop_gain_pips

            else:
                trade_copy.trade_type = TradeType.BUY
                trade_copy.open_price = curr_ao
                trade_copy.stop_loss = curr_ao - trade.pips_risked

                if trade.stop_gain is not None:
                    stop_gain_pips = abs(trade.stop_gain - trade.open_price)
                    trade_copy.stop_gain = curr_ao + stop_gain_pips

            return trade_copy

        else:
            return trade

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        for strategy in self.experts.values():
            self.min_idx = max(self.min_idx, strategy.starting_idx)

        # Safety check
        if curr_idx < self.min_idx:
            return None

        predictions = {}

        for expert_name in self.experts.keys():
            n_samples = self.n_samples[expert_name]

            if n_samples == 0:
                predictions[expert_name] = np.inf

            else:
                empirical_avg = self.empirical_rewards[expert_name] / n_samples
                upper_bound = ((2 * np.log(1 / self.delta)) / n_samples) ** 0.5
                predictions[expert_name] = empirical_avg + upper_bound

        self.expert_name = max(predictions, key=lambda key: predictions[key])
        expert_strategy = self.experts[self.expert_name]
        trade = expert_strategy.place_trade(curr_idx, strategy_data, currency_pair, account_balance)

        if trade is not None:
            n_predictions = 0

            for expert in self.experts.values():
                possible_trade = expert.place_trade(curr_idx, strategy_data, currency_pair, account_balance)
                n_predictions += 1 if possible_trade is not None else 0

            if n_predictions >= self.min_num_predictions:
                self.n_samples[self.expert_name] += 1
                self.use_tsl, self.close_trade_incrementally = \
                    expert_strategy.use_tsl, expert_strategy.close_trade_incrementally

                curr_ao, curr_bo = strategy_data.loc[strategy_data.index[curr_idx], ['Ask_Open', 'Bid_Open']]

                return self._invert(trade, curr_ao, curr_bo)

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
