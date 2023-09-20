from datetime import datetime
from pandas import DataFrame
from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import Trade, TradeType
import numpy as np
from strategies.strategy import Strategy
from typing import Optional


class MarketSimulator(object):
    @staticmethod
    def run_simulation(strategy: Strategy, market_data_raw: DataFrame, strategy_data_raw: DataFrame, currency_pair: str,
                       starting_balance: float = 10000.0) -> MarketSimulationResults:
        # Numerical results we keep track of
        simulation_results = MarketSimulationResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, starting_balance, starting_balance,
                                                     starting_balance, starting_balance, 0, 0)
        pips_risked, curr_trade = [], None
        i = strategy.starting_idx

        # Format the strategies data (each strategies has specific indicators it uses)
        strategy_data = strategy.data_format_function(strategy_data_raw.copy())

        # Helper function for iterating through a trade on the smaller (5 minute) time frame
        def _iterate_through_trade(trade: Trade) -> Optional[datetime]:
            market_data = market_data_raw.loc[market_data_raw['Date'] >= trade.start_date]
            market_data.reset_index(drop=True, inplace=True)

            for j in range(len(market_data)):
                # Check to see if it should close out (there are 4 conditions) or if the market moves in the trade's
                # favor and the strategies decides to move the stop loss
                curr_bid_open, curr_bid_high, curr_bid_low, curr_ask_open, curr_ask_high, curr_ask_low, curr_mid_open, \
                curr_date = market_data.loc[market_data.index[j], ['Bid_Open', 'Bid_High', 'Bid_Low', 'Ask_Open',
                                                                   'Ask_High', 'Ask_Low', 'Mid_Open', 'Date']]

                # Condition 1 - trade is a buy and the stop loss is hit
                if trade.trade_type == TradeType.BUY and curr_bid_low <= trade.stop_loss:
                    trade_amount = (trade.stop_loss - trade.open_price) * trade.n_units
                    day_fees = MarketCalculations.calculate_day_fees(trade, currency_pair, curr_date)
                    simulation_results.update_results(trade_amount, day_fees)

                    return curr_date

                # Condition 2 - Trade is a buy and the take profit/stop gain is hit
                if trade.trade_type == TradeType.BUY and trade.stop_gain is not None and \
                        curr_bid_high >= trade.stop_gain:
                    trade_amount = (trade.stop_gain - trade.open_price) * trade.n_units
                    day_fees = MarketCalculations.calculate_day_fees(trade, currency_pair, curr_date)
                    simulation_results.update_results(trade_amount, day_fees)

                    return curr_date

                # Condition 3 - trade is a sell and the stop loss is hit
                if trade.trade_type == TradeType.SELL and curr_ask_high >= trade.stop_loss:
                    trade_amount = (trade.open_price - trade.stop_loss) * trade.n_units
                    day_fees = MarketCalculations.calculate_day_fees(trade, currency_pair, curr_date)
                    simulation_results.update_results(trade_amount, day_fees)

                    return curr_date

                # Condition 4 - Trade is a sell and the take profit/stop gain is hit
                if trade.trade_type == TradeType.SELL and trade.stop_gain is not None and \
                        curr_ask_low <= trade.stop_gain:
                    trade_amount = (trade.open_price - trade.stop_gain) * trade.n_units
                    day_fees = MarketCalculations.calculate_day_fees(trade, currency_pair, curr_date)
                    simulation_results.update_results(trade_amount, day_fees)

                    return curr_date

                # Check if the strategies decides to move the stop loss - the trade may or may not change
                trade = strategy.move_stop_loss(j, market_data, trade)

                # Check if the strategies decides to close part of the trade - the trade may or may not change, and it
                # might close out completely
                trade = strategy.close_part_of_trade(j, market_data, trade, simulation_results, currency_pair)

                if trade is None:
                    return curr_date

            # If we get to the end of the smaller data frame without returning, that means the simulation is done
            return None

        # Iterate through the strategies data (either on the H4, H1, or M30 time frames)
        while i < len(strategy_data):
            # If there is no open trade, check to see if we should place one
            if curr_trade is None:
                curr_trade = strategy.place_trade(i, strategy_data, currency_pair, simulation_results.account_balance)

            # If a trade was placed, update the simulation results and iterate through the smaller (5-minute) market
            # data to simulate the trade
            if curr_trade is not None:
                # Update the pips risked array
                pips_risked.append(curr_trade.pips_risked)

                # Update the corresponding trade type count
                if curr_trade.trade_type == TradeType.BUY:
                    simulation_results.n_buys += 1

                elif curr_trade.trade_type == TradeType.SELL:
                    simulation_results.n_sells += 1

                else:
                    raise Exception(f'Invalid trade type on the following trade: {curr_trade}')

                # Iterate through the 5-minute data to simulate the trade
                trade_end_date = _iterate_through_trade(curr_trade)
                curr_trade = None

                # If the end date is null, that means we reached the end of the smaller market data, so the simulation
                # is over (so we should break out of the while loop)
                if trade_end_date is None:
                    break

                # Otherwise, update i so that we're on the correct date
                else:
                    i = strategy_data.loc[strategy_data['Date'] <= trade_end_date].index[-1] + 1

            # Otherwise, increment i
            else:
                i += 1

        # Return the simulation results once we've iterated through all the data
        simulation_results.avg_pips_risked = np.array(pips_risked).mean() if len(pips_risked) > 0 else 0

        return simulation_results