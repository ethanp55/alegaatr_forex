from datetime import datetime
from pandas import DataFrame
from market_proxy.market_calculations import MarketCalculations
from market_proxy.market_simulation_results import MarketSimulationResults
from market_proxy.trade import TradeType
import numpy as np
from strategy.strategy import Strategy


class MarketSimulator(object):
    @staticmethod
    def run_simulation(strategy: Strategy, market_data: DataFrame, currency_pair: str,
                       starting_balance: float = 10000.0) -> MarketSimulationResults:
        market_data = strategy.data_format_function(market_data)
        reward, n_wins, n_losses, win_streak, loss_streak, curr_win_streak, curr_loss_streak, n_buys, n_sells, \
        day_fees = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Numerical results we keep track of
        pips_risked, trade = [], None
        account_balance, lowest_account_balance, highest_account_balance = \
            starting_balance, starting_balance, starting_balance

        for idx in range(strategy.starting_idx, len(market_data)):
            # If there is no open trade, check to see if we should place one
            if trade is None:
                trade = strategy.place_trade(idx, market_data, currency_pair, account_balance)

                if trade is not None:
                    pips_risked.append(trade.pips_risked)

                    if trade.trade_type == TradeType.BUY:
                        n_buys += 1

                    elif trade.trade_type == TradeType.SELL:
                        n_sells += 1

                    else:
                        raise Exception(f'Invalid trade type on the following trade: {trade}')

            # If there is an open trade, check to see if it should close out (there are 4 conditions) or if the
            # market moves in the trade's favor and the strategy decides to move the stop loss
            if trade is not None:
                curr_bid_open, curr_bid_high, curr_bid_low, curr_ask_open, curr_ask_high, curr_ask_low, curr_mid_open, \
                curr_date = market_data.loc[market_data.index[idx], ['Bid_Open', 'Bid_High', 'Bid_Low', 'Ask_Open',
                                                                     'Ask_High', 'Ask_Low', 'Mid_Open', 'Date']]

                # Condition 1 - trade is a buy and the stop loss is hit
                if trade.trade_type == TradeType.BUY and curr_bid_low <= trade.stop_loss:
                    trade.end_date = datetime.strptime(curr_date, '%Y-%m-%d %H:%M:%S')
                    trade_amount = (trade.stop_loss - trade.open_price) * trade.n_units
                    reward += trade_amount
                    day_fees += MarketCalculations.calculate_day_fees(trade, currency_pair)
                    account_balance += reward + day_fees
                    lowest_account_balance = min(lowest_account_balance, account_balance)
                    highest_account_balance = max(highest_account_balance, account_balance)

                    n_wins += 1 if trade_amount > 0 else 0
                    n_losses += 1 if trade_amount < 0 else 0
                    curr_win_streak = 0 if trade_amount < 0 else curr_win_streak + 1
                    curr_loss_streak = 0 if trade_amount > 0 else curr_loss_streak + 1

                    if curr_win_streak > win_streak:
                        win_streak = curr_win_streak

                    if curr_loss_streak > loss_streak:
                        loss_streak = curr_loss_streak

                    trade = None

                    continue

                # Condition 2 - Trade is a buy and the take profit/stop gain is hit
                if trade.trade_type == TradeType.BUY and trade.stop_gain is not None and \
                        curr_bid_high >= trade.stop_gain:
                    trade.end_date = datetime.strptime(curr_date, '%Y-%m-%d %H:%M:%S')
                    trade_amount = (trade.stop_gain - trade.open_price) * trade.n_units
                    reward += trade_amount
                    day_fees += MarketCalculations.calculate_day_fees(trade, currency_pair)
                    account_balance += reward + day_fees
                    lowest_account_balance = min(lowest_account_balance, account_balance)
                    highest_account_balance = max(highest_account_balance, account_balance)

                    n_wins += 1 if trade_amount > 0 else 0
                    n_losses += 1 if trade_amount < 0 else 0
                    curr_win_streak = 0 if trade_amount < 0 else curr_win_streak + 1
                    curr_loss_streak = 0 if trade_amount > 0 else curr_loss_streak + 1

                    if curr_win_streak > win_streak:
                        win_streak = curr_win_streak

                    if curr_loss_streak > loss_streak:
                        loss_streak = curr_loss_streak

                    trade = None

                    continue

                # Condition 3 - trade is a sell and the stop loss is hit
                if trade.trade_type == TradeType.SELL and curr_ask_high >= trade.stop_loss:
                    trade.end_date = datetime.strptime(curr_date, '%Y-%m-%d %H:%M:%S')
                    trade_amount = (trade.open_price - trade.stop_loss) * trade.n_units
                    reward += trade_amount
                    day_fees += MarketCalculations.calculate_day_fees(trade, currency_pair)
                    account_balance += reward + day_fees
                    lowest_account_balance = min(lowest_account_balance, account_balance)
                    highest_account_balance = max(highest_account_balance, account_balance)

                    n_wins += 1 if trade_amount > 0 else 0
                    n_losses += 1 if trade_amount < 0 else 0
                    curr_win_streak = 0 if trade_amount < 0 else curr_win_streak + 1
                    curr_loss_streak = 0 if trade_amount > 0 else curr_loss_streak + 1

                    if curr_win_streak > win_streak:
                        win_streak = curr_win_streak

                    if curr_loss_streak > loss_streak:
                        loss_streak = curr_loss_streak

                    trade = None

                    continue

                # Condition 4 - Trade is a sell and the take profit/stop gain is hit
                if trade.trade_type == TradeType.SELL and trade.stop_gain is not None and \
                        curr_ask_low <= trade.stop_gain:
                    trade.end_date = datetime.strptime(curr_date, '%Y-%m-%d %H:%M:%S')
                    trade_amount = (trade.open_price - trade.stop_gain) * trade.n_units
                    reward += trade_amount
                    day_fees += MarketCalculations.calculate_day_fees(trade, currency_pair)
                    account_balance += reward + day_fees
                    lowest_account_balance = min(lowest_account_balance, account_balance)
                    highest_account_balance = max(highest_account_balance, account_balance)

                    n_wins += 1 if trade_amount > 0 else 0
                    n_losses += 1 if trade_amount < 0 else 0
                    curr_win_streak = 0 if trade_amount < 0 else curr_win_streak + 1
                    curr_loss_streak = 0 if trade_amount > 0 else curr_loss_streak + 1

                    if curr_win_streak > win_streak:
                        win_streak = curr_win_streak

                    if curr_loss_streak > loss_streak:
                        loss_streak = curr_loss_streak

                    trade = None

                    continue

                # Check if the strategy decides to move the stop loss - the trade may or may not change
                trade = strategy.move_stop_loss(idx, market_data, trade)

        # Return the simulation results once we've iterated through all the data
        avg_pips_risked = np.array(pips_risked).mean() if len(pips_risked) > 0 else np.nan

        return MarketSimulationResults(reward, day_fees, reward + day_fees, avg_pips_risked, n_buys, n_sells, n_wins,
                                       n_losses, win_streak, loss_streak, lowest_account_balance,
                                       highest_account_balance)
