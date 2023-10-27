from aat.assumptions import Assumptions
from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import Trade, TradeType
import numpy as np
from pandas import DataFrame
import pickle
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


class AlegAATr(Strategy):
    def __init__(self, starting_idx: int = 2, percent_to_risk: float = 0.02, min_num_predictions: int = 3,
                 use_single_selection: bool = True, min_n_neighbors: int = 15) -> None:
        super().__init__(starting_idx, percent_to_risk, 'AlegAATr')
        self.generators = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(),
                           MACD(), MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(),
                           Supertrend()]
        self.models, self.correction_terms = {}, {}
        self.min_num_predictions, self.use_single_selection, self.min_n_neighbors = \
            min_num_predictions, use_single_selection, min_n_neighbors
        self.use_tsl, self.close_trade_incrementally = False, False
        self.min_idx = 0
        self.prev_prediction = None
        self.predictions_when_wrong, self.trade_values_when_wrong = [], []
        self.predictions_when_correct, self.trade_values_when_correct = [], []

    def print_parameters(self) -> None:
        print(f'min_num_predictions: {self.min_num_predictions}')
        print(f'use_single_selection: {self.use_single_selection}')

    def _clear_metric_tracking_vars(self) -> None:
        self.prev_prediction = None
        self.predictions_when_wrong.clear()
        self.trade_values_when_wrong.clear()
        self.predictions_when_correct.clear()
        self.trade_values_when_correct.clear()

    def update_metric_tracking_vars(self, trade_value: float) -> None:
        # Win
        if trade_value > 0:
            self.predictions_when_correct.append(self.prev_prediction)
            self.trade_values_when_correct.append(trade_value)

        # Loss
        else:
            self.predictions_when_wrong.append(self.prev_prediction)
            self.trade_values_when_wrong.append(trade_value)

        self.prev_prediction = None

    def save_metric_tracking_vars(self, currency_pair: str, time_frame: str) -> None:
        file_path = f'../experiments/results/alegaatr_metrics/{currency_pair}_{time_frame}'

        with open(f'{file_path}_predictions_when_wrong.pickle', 'wb') as f:
            pickle.dump(self.predictions_when_wrong, f)

        with open(f'{file_path}_trade_values_when_wrong.pickle', 'wb') as f:
            pickle.dump(self.trade_values_when_wrong, f)

        with open(f'{file_path}_predictions_when_correct.pickle', 'wb') as f:
            pickle.dump(self.predictions_when_correct, f)

        with open(f'{file_path}_trade_values_when_correct.pickle', 'wb') as f:
            pickle.dump(self.trade_values_when_correct, f)

        self._clear_metric_tracking_vars()

    def load_best_parameters(self, currency_pair: str, time_frame: str) -> None:
        try:
            super().load_best_parameters(currency_pair, time_frame)

        except:
            pass

        self.models, self.correction_terms = {}, {}

        for generator in self.generators:
            try:
                # Make sure the generator is using its "best" parameters for the given currency pair and time frame
                generator.load_best_parameters(currency_pair, time_frame)

                # Load the KNN model and correction terms for the generator on the given currency pair and time frame
                strategy_name = generator.name
                name_pair_time_str = f'{strategy_name}_{currency_pair}_{time_frame}'
                correction_terms_file_name = f'../aat/training_data/{name_pair_time_str}_aat_correction_terms.pickle'
                knn_file_name = f'../aat/training_data/{name_pair_time_str}_aat_knn.pickle'
                self.models[strategy_name] = pickle.load(open(knn_file_name, 'rb'))
                self.correction_terms[strategy_name] = pickle.load(open(correction_terms_file_name, 'rb'))

            except:
                continue

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        for generator in self.generators:
            self.min_idx = max(self.min_idx, generator.starting_idx)

        # Safety check
        if curr_idx < self.min_idx:
            return None

        new_assumptions = Assumptions(strategy_data, curr_idx, currency_pair, 0.0)
        x = np.array(new_assumptions.create_aat_tuple()[:-1], dtype=float).reshape(1, -1)

        if self.use_single_selection:
            return self._single_selection(x, curr_idx, strategy_data, currency_pair, account_balance)

        else:
            return self._weighted_ensemble(x, curr_idx, strategy_data, currency_pair, account_balance)

    def _weighted_ensemble(self, x: np.array, curr_idx: int, strategy_data: DataFrame, currency_pair: str,
                           account_balance: float) -> Optional[Trade]:
        trades, amount_predictions = [], []

        for generator in self.generators:
            trade = generator.place_trade(curr_idx, strategy_data, currency_pair, account_balance)

            if trade is not None and generator.name in self.models:
                knn_model, training_data = self.models[generator.name], self.correction_terms[generator.name]
                n_neighbors = len(training_data)

                if n_neighbors >= self.min_n_neighbors:
                    neighbor_distances, neighbor_indices = knn_model.kneighbors(x, n_neighbors)
                    corrections, distances = [], []
                    baseline = abs(trade.open_price - trade.stop_loss) * trade.n_units

                    for i in range(len(neighbor_indices[0])):
                        neighbor_idx = neighbor_indices[0][i]
                        neighbor_dist = neighbor_distances[0][i]
                        corrections.append(training_data[neighbor_idx,])
                        distances.append(neighbor_dist)

                    trade_amount_pred, inverse_distance_sum = 0, 0

                    for dist in distances:
                        inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)

                    for i in range(len(corrections)):
                        distance_i, cor = distances[i], corrections[i]
                        inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
                        distance_weight = inverse_distance_i / inverse_distance_sum

                        trade_amount_pred += (baseline * cor * distance_weight)

                    if trade_amount_pred > 0:
                        trades.append((trade, generator))
                        amount_predictions.append(trade_amount_pred)

        if len(trades) < self.min_num_predictions:
            return None

        assert len(trades) == len(amount_predictions)

        amount_predictions_sum = sum(amount_predictions)

        # Keep track of different metrics that the strategies "vote" on
        n_buys, n_sells = 0, 0
        n_buy_use_tsl, n_sell_use_tsl, n_buy_close_trade_incrementally, n_sell_close_trade_incrementally = 0, 0, 0, 0
        buy_sl_pips, sell_sl_pips = [], []
        buy_sg_pips, sell_sg_pips = [], []
        n_buy_sg, n_sell_sg = 0, 0

        # Current data used for trade calculations
        curr_date, curr_ao, curr_bo, curr_mo, curr_bh, curr_al = strategy_data.loc[
            strategy_data.index[curr_idx], ['Date', 'Ask_Open', 'Bid_Open', 'Mid_Open', 'Bid_High', 'Ask_Low']]
        spread = abs(curr_ao - curr_bo)

        # Predicted trade value
        trade_value = np.array(amount_predictions).mean()

        for i in range(len(trades)):
            (trade, generator), amount_prediction = trades[i], amount_predictions[i]
            amount_prediction_weight = amount_prediction / amount_predictions_sum

            if trade.trade_type == TradeType.BUY:
                n_buys += amount_prediction_weight
                n_buy_use_tsl += amount_prediction_weight if (
                        hasattr(generator, 'use_tsl') and generator.use_tsl) else 0
                n_buy_close_trade_incrementally += amount_prediction_weight if \
                    (hasattr(generator, 'close_trade_incrementally') and generator.close_trade_incrementally) else 0
                buy_sl_pips.append(trade.pips_risked)

                if trade.stop_gain is not None:
                    n_buy_sg += amount_prediction_weight
                    buy_sg_pips.append(trade.stop_gain - curr_ao)

            else:
                n_sells += amount_prediction_weight
                n_sell_use_tsl += amount_prediction_weight if (
                        hasattr(generator, 'use_tsl') and generator.use_tsl) else 0
                n_sell_close_trade_incrementally += amount_prediction_weight if \
                    (hasattr(generator, 'close_trade_incrementally') and generator.close_trade_incrementally) else 0
                sell_sl_pips.append(trade.pips_risked)

                if trade.stop_gain is not None:
                    n_sell_sg += amount_prediction_weight
                    sell_sg_pips.append(curr_bo - trade.stop_gain)

        # If there are more buy votes than sell votes and the number of buy votes meets the minimum number of votes,
        # place a buy
        if n_buys > n_sells:
            open_price = curr_ao
            sl_pips = sum(buy_sl_pips) / len(buy_sl_pips)
            stop_loss = open_price - sl_pips

            if stop_loss < open_price and spread <= sl_pips * 0.1:
                trade_type = TradeType.BUY
                amount_to_risk = account_balance * self.percent_to_risk
                n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                         currency_pair, amount_to_risk)
                stop_gain = None if n_buy_sg < n_buys * 0.5 else open_price + (
                        sum(buy_sg_pips) / len(buy_sg_pips))

                self.use_tsl = n_buy_use_tsl >= n_buys * 0.5
                self.close_trade_incrementally = n_buy_close_trade_incrementally >= n_buys * 0.5
                self.prev_prediction = trade_value

                return Trade(trade_type, open_price, stop_loss, stop_gain, n_units, sl_pips, curr_date,
                             currency_pair)

        # If there are more sell votes than buy votes and the number of sell votes meets the minimum number of votes,
        # place a sell
        elif n_sells > n_buys:
            open_price = curr_bo
            sl_pips = sum(sell_sl_pips) / len(sell_sl_pips)
            stop_loss = open_price + sl_pips

            if stop_loss > open_price and spread <= sl_pips * 0.1:
                trade_type = TradeType.SELL
                amount_to_risk = account_balance * self.percent_to_risk
                n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                         currency_pair, amount_to_risk)
                stop_gain = None if n_sell_sg < n_sells * 0.5 else open_price - (
                        sum(sell_sg_pips) / len(sell_sg_pips))

                self.use_tsl = n_sell_use_tsl >= n_sells * 0.5
                self.close_trade_incrementally = n_sell_close_trade_incrementally >= n_sells * 0.5
                self.prev_prediction = trade_value

                return Trade(trade_type, open_price, stop_loss, stop_gain, n_units, sl_pips, curr_date,
                             currency_pair)

        return None

    def _single_selection(self, x: np.array, curr_idx: int, strategy_data: DataFrame, currency_pair: str,
                          account_balance: float) -> Optional[Trade]:
        best_trade, best_trade_amount, n_profitable_predictions = None, -np.inf, 0

        for generator in self.generators:
            trade = generator.place_trade(curr_idx, strategy_data, currency_pair, account_balance)

            if trade is not None and generator.name in self.models:
                knn_model, training_data = self.models[generator.name], self.correction_terms[generator.name]
                n_neighbors = len(training_data)

                if n_neighbors >= self.min_n_neighbors:
                    neighbor_distances, neighbor_indices = knn_model.kneighbors(x, n_neighbors)
                    corrections, distances = [], []
                    baseline = abs(trade.open_price - trade.stop_loss) * trade.n_units

                    for i in range(len(neighbor_indices[0])):
                        neighbor_idx = neighbor_indices[0][i]
                        neighbor_dist = neighbor_distances[0][i]
                        corrections.append(training_data[neighbor_idx,])
                        distances.append(neighbor_dist)

                    trade_amount_pred, inverse_distance_sum = 0, 0

                    for dist in distances:
                        inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)

                    for i in range(len(corrections)):
                        distance_i, cor = distances[i], corrections[i]
                        inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
                        distance_weight = inverse_distance_i / inverse_distance_sum

                        trade_amount_pred += (baseline * cor * distance_weight)

                    n_profitable_predictions += 1 if trade_amount_pred > 0 else 0

                    if trade_amount_pred > max(0, best_trade_amount):
                        best_trade, best_trade_amount = trade, trade_amount_pred
                        self.use_tsl, self.close_trade_incrementally = \
                            generator.use_tsl, generator.close_trade_incrementally

        if n_profitable_predictions >= self.min_num_predictions:
            self.prev_prediction = best_trade_amount
            return best_trade

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
