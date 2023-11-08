import os
import numpy as np
import pandas as pd
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES


def crunch_numbers() -> None:
    def total_profit() -> None:
        all_profits, m30_profits, h1_profits, h4_profits = {}, {}, {}, {}
        all_profit_sums, m30_profit_sums, h1_profit_sums, h4_profit_sums = {}, {}, {}, {}
        directory = '../experiments/results/final_balances_csv/'
        file_list = os.listdir(directory)

        pair_time_frame_combos = [f'{currency_pair}_{time_frame}' for currency_pair in CURRENCY_PAIRS for time_frame in
                                  TIME_FRAMES]
        results_by_pair_time_frame = {}

        for pair_time_frame in pair_time_frame_combos:
            results_by_pair_time_frame[pair_time_frame] = {}

        # Extract the data
        for file_name in file_list:
            df = pd.read_csv(f'{directory}{file_name}')

            for strategy in df.columns[1:]:
                profit = df.loc[df.index[0], strategy] - 10000
                all_profits[strategy] = all_profits.get(strategy, []) + [profit]
                all_profit_sums[strategy] = all_profit_sums.get(strategy, 0) + profit

                if 'M30' in file_name:
                    m30_profits[strategy] = m30_profits.get(strategy, []) + [profit]
                    m30_profit_sums[strategy] = m30_profit_sums.get(strategy, 0) + profit

                elif 'H1' in file_name:
                    h1_profits[strategy] = h1_profits.get(strategy, []) + [profit]
                    h1_profit_sums[strategy] = h1_profit_sums.get(strategy, 0) + profit

                else:
                    h4_profits[strategy] = h4_profits.get(strategy, []) + [profit]
                    h4_profit_sums[strategy] = h4_profit_sums.get(strategy, 0) + profit

            for pair_time_frame in results_by_pair_time_frame.keys():
                if pair_time_frame in file_name:
                    for strategy in df.columns[1:]:
                        profit = df.loc[df.index[0], strategy] - 10000
                        results_by_pair_time_frame[pair_time_frame][strategy] = results_by_pair_time_frame[
                                                                                    pair_time_frame].get(strategy,
                                                                                                         []) + [profit]

        # Create tuples for each strategy and profit sum
        profit_with_names = [(strategy, profit) for strategy, profit in all_profit_sums.items()]
        m30_with_names = [(strategy, profit) for strategy, profit in m30_profit_sums.items()]
        h1_with_names = [(strategy, profit) for strategy, profit in h1_profit_sums.items()]
        h4_with_names = [(strategy, profit) for strategy, profit in h4_profit_sums.items()]

        # Sort the results so that the most profitable results are first
        profit_with_names.sort(key=lambda x: x[1], reverse=True)
        m30_with_names.sort(key=lambda x: x[1], reverse=True)
        h1_with_names.sort(key=lambda x: x[1], reverse=True)
        h4_with_names.sort(key=lambda x: x[1], reverse=True)

        # Print the profit sums
        for time_frame in TIME_FRAMES:
            sum_to_print = m30_with_names if time_frame == 'M30' else (
                h1_with_names if time_frame == 'H1' else h4_with_names)

            print(f'PROFIT SUMS FOR {time_frame}:')

            for strategy, profit in sum_to_print:
                print(f'{strategy}\'s total profit: {profit}')

            print()

        print('PROFIT SUM ACROSS EVERY CATEGORY')

        for strategy, profit in profit_with_names:
            print(f'{strategy}\'s total profit: {profit}')

        print()

        # Print the profit results for each currency and time frame pair
        for pair_time_frame in results_by_pair_time_frame.keys():
            print(f'PAIR-TIME RESULTS FOR {pair_time_frame}:')

            for strategy, profits in results_by_pair_time_frame[pair_time_frame].items():
                profits_array = np.array(profits)

                print(f'{strategy}\'s avg profit: {profits_array.mean()}, med profit: {np.median(profits_array)}, '
                      f'profit std: {profits_array.std()}, min: {profits_array.min()}, max: {profits_array.max()}, '
                      f'profit sum: {profits_array.sum()}')

            print()

        # Print the profit results for each time frame
        for time_frame in TIME_FRAMES:
            profits_to_print = m30_profits if time_frame == 'M30' else (
                h1_profits if time_frame == 'H1' else h4_profits)

            print(f'PROFIT RESULTS FOR {time_frame}:')

            for strategy, profits in profits_to_print.items():
                profits_array = np.array(profits)

                print(f'{strategy}\'s avg profit: {profits_array.mean()}, med profit: {np.median(profits_array)}, '
                      f'profit std: {profits_array.std()}, min: {profits_array.min()}, max: {profits_array.max()}')

            print()

        for strategy, profits in all_profits.items():
            profits_array = np.array(profits)
            print(f'{strategy}\'s avg profit: {profits_array.mean()}, med profit: {np.median(profits_array)}, '
                  f'profit std: {profits_array.std()}, min: {profits_array.min()}, max: {profits_array.max()}')

    def profitable_ratios() -> None:
        # Print out the profit ratio of the test sets
        directory = '../experiments/results/ratios_csv/'
        file_list = os.listdir(directory)
        ratios, ratios_with_names = [], []

        for file_name in file_list:
            df = pd.read_csv(f'{directory}{file_name}')

            for strategy in df.columns[1:]:
                profit_ratio = df.loc[df.index[0], strategy]
                ratios.append(profit_ratio)
                ratios_with_names.append((strategy, profit_ratio))

        ratios_with_names.sort(key=lambda x: x[1], reverse=True)

        for strategy, ratio in ratios_with_names:
            print(f'{strategy}\'s profitable ratio: {ratio}')

        print(f'\nAverage profitable ratio: {np.array(ratios).mean()}, med profitable ratio: {np.median(ratios)}')

    # Calculate profit sums
    total_profit()

    print()

    # Print out profit ratios
    profitable_ratios()


if __name__ == "__main__":
    crunch_numbers()
