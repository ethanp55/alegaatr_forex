from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import Trade
import numpy as np
from pandas import DataFrame
import random
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


class EXP3(Strategy):
    def __init__(self, starting_idx: int = 1, percent_to_risk: float = 0.02, gamma: float = 0.5) -> None:
        super().__init__(starting_idx, percent_to_risk, 'EXP3')
        strategy_pool = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(),
                         MACD(), MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(),
                         Supertrend()]
        self.experts, self.empirical_rewards, self.weights, self.expert_name, self.p_t_vals = {}, {}, {}, None, None
        self.use_tsl, self.close_trade_incrementally = False, False
        self.min_idx = 0

        for expert in strategy_pool:
            self.experts[expert.name] = expert
            self.empirical_rewards[expert.name] = 0
            self.weights[expert.name] = 1.0

        self.gamma, self.k = gamma, len(self.experts)

    def trade_finished(self, net_profit: float) -> None:
        expert_names = list(self.weights.keys())
        x_hat = net_profit / self.p_t_vals[expert_names.index(self.expert_name)]
        self.weights[self.expert_name] = self.weights[self.expert_name] * np.exp((self.gamma * x_hat) / self.k)

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        for strategy in self.experts.values():
            self.min_idx = max(self.min_idx, strategy.starting_idx)

        # Safety check
        if curr_idx < self.min_idx:
            return None

        weight_sum = sum(self.weights.values())
        self.p_t_vals = [((1 - self.gamma) * (weight / weight_sum)) + (self.gamma / self.k) for weight in
                         self.weights.values()]
        self.expert_name = random.choices(list(self.weights.keys()), weights=self.p_t_vals, k=1)[0]
        expert = self.experts[self.expert_name]
        self.use_tsl, self.close_trade_incrementally = expert.use_tsl, expert.close_trade_incrementally

        return expert.place_trade(curr_idx, strategy_data, currency_pair, account_balance)

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