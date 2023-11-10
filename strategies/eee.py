from copy import deepcopy
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import Trade, TradeType
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


class EEE(Strategy):
    def __init__(self, starting_idx: int = 1, percent_to_risk: float = 0.02, explore_prob: float = 0.1,
                 genetic: bool = False, min_num_predictions: int = 1, invert: bool = False) -> None:
        super().__init__(starting_idx, percent_to_risk, 'EEE')
        strategy_pool = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(),
                         MACD(), MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(),
                         Supertrend()]
        self.experts = {}

        for strategy in strategy_pool:
            self.experts[strategy.name] = strategy

        self.min_idx = 0
        self.m_e, self.n_e, self.s_e = {}, {}, {}
        self.in_phase, self.phase_counter, self.phase_rewards, self.n_i = False, 0, [], 0
        self.agent_in_use = random.choice(list(self.experts.values()))
        self.use_tsl, self.close_trade_incrementally = \
            self.agent_in_use.use_tsl, self.agent_in_use.close_trade_incrementally
        self.explore_prob, self.genetic, self.min_num_predictions, self.invert = \
            explore_prob, genetic, min_num_predictions, invert

        for agent_name in self.experts.keys():
            self.m_e[agent_name] = 0
            self.n_e[agent_name] = 0
            self.s_e[agent_name] = 0

    def load_best_parameters(self, currency_pair: str, time_frame: str, year: int) -> None:
        if not self.genetic:
            super().load_best_parameters(currency_pair, time_frame, year)

        for generator in self.experts:
            try:
                # Make sure the generator is using its "best" parameters for the given currency pair and time frame
                generator.load_best_parameters(currency_pair, time_frame, year)

            except:
                continue

    def trade_finished(self, net_profit: float) -> None:
        self.phase_rewards.append(net_profit)

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

        if self.in_phase:
            if self.phase_counter < self.n_i:
                self.phase_counter += 1

            else:
                avg_phase_reward = np.array(self.phase_rewards).mean() if len(self.phase_rewards) > 0 else 0
                self.n_e[self.agent_in_use.name] += 1
                self.s_e[self.agent_in_use.name] += self.n_i
                self.m_e[self.agent_in_use.name] = self.m_e[self.agent_in_use.name] + (
                        self.n_i / self.s_e[self.agent_in_use.name]) * (
                                                           avg_phase_reward - self.m_e[self.agent_in_use.name])
                self.phase_rewards, self.phase_counter, self.n_i, self.in_phase = [], 0, 0, False

        if not self.in_phase:
            explore = np.random.choice([0, 1], p=[1 - self.explore_prob, self.explore_prob])

            if explore:
                new_agent = random.choice(list(self.experts.values()))

                self.agent_in_use = new_agent
                self.use_tsl, self.close_trade_incrementally = \
                    self.agent_in_use.use_tsl, self.agent_in_use.close_trade_incrementally

            else:
                max_reward, agents_to_consider = max(list(self.m_e.values())), []

                for key, val in self.m_e.items():
                    if val == max_reward:
                        agents_to_consider.append(self.experts[key])

                new_agent = random.choice(agents_to_consider)

                self.agent_in_use = new_agent
                self.use_tsl, self.close_trade_incrementally = \
                    self.agent_in_use.use_tsl, self.agent_in_use.close_trade_incrementally

            self.n_i, self.in_phase = np.random.choice(list(range(1, 50))), True

        trade = self.agent_in_use.place_trade(curr_idx, strategy_data, currency_pair, account_balance)

        if trade is not None:
            n_predictions = 0

            for expert in self.experts:
                possible_trade = expert.place_trade(curr_idx, strategy_data, currency_pair, account_balance)
                n_predictions += 1 if possible_trade is not None else 0

            if n_predictions >= self.min_num_predictions:
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
