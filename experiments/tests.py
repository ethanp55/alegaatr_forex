import os
import numpy as np
import pandas as pd
import pickle
from statsmodels.stats.multicomp import MultiComparison
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def run_tests() -> None:
    def _test_trade_amounts() -> None:
        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[1:]:
                    pair_time_year_str = f'{currency_pair}_{time_frame}_{year}'
                    file_list = os.listdir('../experiments/results/')
                    filtered_file_list = [file_name for file_name in file_list if
                                          (pair_time_year_str in file_name and 'trade_amounts' in file_name)]

                    trade_amounts, strategy_names = [], []
                    avg_amounts, win_rates, names_to_save = [], [], []

                    for file_name in filtered_file_list:
                        strategy_name = file_name.split('_')[0]
                        amounts = pickle.load(open(f'../experiments/results/{file_name}', 'rb'))

                        for amount in amounts:
                            strategy_names.append(strategy_name)
                            trade_amounts.append(amount)

                        # Save the win rate and average trade amount
                        amounts_array = np.array(amounts)
                        win_rate = np.sum(amounts_array > 0) / len(amounts)
                        avg_trade_amount = amounts_array.mean()
                        avg_amounts.append(avg_trade_amount)
                        win_rates.append(win_rate)
                        names_to_save.append(strategy_name)

                    # Export the win rates and average trade amounts as csv files
                    df = pd.DataFrame([win_rates], columns=names_to_save)
                    df.to_csv(f'../experiments/results/trade_amounts_csv/{pair_time_year_str}_win_rate.csv')
                    df = pd.DataFrame([avg_amounts], columns=names_to_save)
                    df.to_csv(f'../experiments/results/trade_amounts_csv/{pair_time_year_str}_avg_amounts.csv')

                    # Perform Tukey-Kramer multi-comparison test
                    data = {
                        'strategy': strategy_names,
                        'amounts': trade_amounts
                    }
                    df = pd.DataFrame(data)
                    mc = MultiComparison(df['amounts'], df['strategy'])
                    result = mc.tukeyhsd()

                    print(f'TUKEY KRAMER ON {pair_time_year_str}')
                    print(result)
                    print()

    def _extract_profitable_ratios() -> None:
        file_list = os.listdir('../experiments/results/')
        filtered_file_list = [file_name for file_name in file_list if 'profitable_ratios' in file_name]

        ratios, strategy_names = [], []

        for file_name in filtered_file_list:
            strategy_name = file_name.split('_')[0]
            ratio = pickle.load(open(f'../experiments/results/{file_name}', 'rb'))

            ratios.append(ratio)
            strategy_names.append(strategy_name)

        df = pd.DataFrame([ratios], columns=strategy_names)
        df.to_csv(f'../experiments/results/ratios_csv/profitable_ratios.csv')

    def _test_profits() -> None:
        profits_by_strategy = {}
        file_list = os.listdir('../experiments/results/')
        filtered_file_list = [file_name for file_name in file_list if
                              ('final_balances' in file_name and 'csv' not in file_name)]

        for file_name in filtered_file_list:
            strategy_name = file_name.split('_')[0]
            profit = float(pickle.load(open(f'../experiments/results/{file_name}', 'rb')) - 10000)

            profits_by_strategy[strategy_name] = profits_by_strategy.get(strategy_name, []) + [
                profit]

        data, groups = [], []

        for strategy, profits in profits_by_strategy.items():
            data.extend(profits)
            groups.extend([strategy] * len(profits))

        mc = MultiComparison(data, groups)
        result = mc.tukeyhsd()

        print(result.summary())

    def _test_bandit_profits() -> None:
        profits_by_strategy = {}
        file_list = os.listdir('../experiments/results/')
        filtered_file_list = [file_name for file_name in file_list if
                              ('final_balances' in file_name and 'csv' not in file_name)]
        bandit_strategy_names = ['AlegAATr', 'UCB', 'EXP3', 'EEE']

        for file_name in filtered_file_list:
            strategy_name = file_name.split('_')[0]

            if strategy_name in bandit_strategy_names:
                profit = float(pickle.load(open(f'../experiments/results/{file_name}', 'rb')) - 10000)

                profits_by_strategy[strategy_name] = profits_by_strategy.get(strategy_name, []) + [
                    profit]

        data, groups = [], []

        for strategy, profits in profits_by_strategy.items():
            data.extend(profits)
            groups.extend([strategy] * len(profits))

        mc = MultiComparison(data, groups)
        result = mc.tukeyhsd()

        print(result.summary())

    # Extract info about trade amounts and run multi-comparison tests
    # _test_trade_amounts()

    # Extract the overall profitable ratios for each strategy
    _extract_profitable_ratios()

    # Run multi-comparison test on final profit amounts
    _test_profits()

    # Run multi-comparison test on final profit amounts
    _test_bandit_profits()


if __name__ == "__main__":
    run_tests()
