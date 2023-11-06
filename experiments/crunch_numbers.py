import os
import numpy as np
import pandas as pd
from utils.utils import TIME_FRAMES


def crunch_numbers() -> None:
    def total_profit() -> None:
        # Keep track of profit sums for each strategy across the different time frames
        all_profits, m30_profits, h1_profits, h4_profits = {}, {}, {}, {}
        all_profit_sums, m30_profit_sums, h1_profit_sums, h4_profit_sums = {}, {}, {}, {}
        directory = '../experiments/results/final_balances_csv/'
        file_list = os.listdir(directory)

        # Calculate the profit sums
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

        # Print the results
        for time_frame in TIME_FRAMES:
            sum_to_print = m30_with_names if time_frame == 'M30' else (
                h1_with_names if time_frame == 'H1' else h4_with_names)

            print(f'TOTAL PROFIT SUMS FOR {time_frame}:')

            for strategy, profit in sum_to_print:
                print(f'{strategy}\'s total profit: {profit}')

            print()

        print('PROFIT ACROSS EVERY CATEGORY')

        for strategy, profit in profit_with_names:
            print(f'{strategy}\'s total profit: {profit}')

        for time_frame in TIME_FRAMES:
            profits_to_print = m30_profits if time_frame == 'M30' else (
                h1_profits if time_frame == 'H1' else h4_profits)

            print(f'TOTAL PROFIT RESULTS FOR {time_frame}:')

            for strategy, profits in profits_to_print:
                profits_array = np.array(profits)

                print(f'{strategy}\'s avg profit: {profits_array.mean()}, profit std: {profits_array.std()}')

            print()

        for strategy, profits in all_profits:
            profits_array = np.array(profits)
            print(f'{strategy}\'s avg profit: {profits_array.mean()}, profit std: {profits_array.std()}')

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

        print(f'\nAverage profitable ratio: {np.array(ratios).mean()}')

    # Calculate profit sums
    total_profit()

    print()

    # Print out profit ratios
    profitable_ratios()


if __name__ == "__main__":
    crunch_numbers()
