from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES


def create_plots() -> None:
    def _create_account_value_plots() -> None:
        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                pair_time_str = f'{currency_pair}_{time_frame}'
                file_list = os.listdir('../experiments/results/')
                filtered_file_list = [file_name for file_name in file_list if
                                      (pair_time_str in file_name and 'account_values' in file_name)]

                account_values, strategy_names, max_len = [], [], 0

                for file_name in filtered_file_list:
                    strategy_name = file_name.split('_')[0]
                    values = pickle.load(open(f'../experiments/results/{file_name}', 'rb'))

                    strategy_names.append(strategy_name)
                    account_values.append(values)
                    max_len = max(max_len, len(values))

                    # Plot for each strategy
                    x = range(0, len(values))
                    plt.plot(x, values)
                    plt.xlabel('Time')
                    plt.ylabel('Account Balance')
                    plt.title(f'Account Balances Over Time For {strategy_name}')
                    plt.savefig(f'../experiments/plots/{strategy_name}_{pair_time_str}_account_values',
                                bbox_inches='tight')
                    plt.clf()

                # Plot containing all strategies
                x = list(range(0, max_len))
                x_common = np.array(x)

                for i in range(len(account_values)):
                    values, strategy_name = account_values[i], strategy_names[i]

                    if len(values) < max_len:
                        x = list(range(0, len(values)))
                        values_interp = np.interp(x_common, x, values)

                    else:
                        values_interp = values

                    plt.plot(x_common, values_interp, label=strategy_name)

                plt.xlabel('Time')
                plt.ylabel('Account Balance')
                plt.title('Account Balances Over Time')
                plt.legend(loc='best')
                plt.savefig(f'../experiments/plots/{pair_time_str}_account_values', bbox_inches='tight')
                plt.clf()

    def _create_final_balance_bar_graphs() -> None:
        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                pair_time_str = f'{currency_pair}_{time_frame}'
                file_list = os.listdir('../experiments/results/')
                filtered_file_list = [file_name for file_name in file_list if
                                      (pair_time_str in file_name and 'final_balances' in file_name)]

                final_balances, strategy_names, max_len = [], [], 0

                for file_name in filtered_file_list:
                    strategy_name = file_name.split('_')[0]
                    final_balance = pickle.load(open(f'../experiments/results/{file_name}', 'rb'))

                    strategy_names.append(strategy_name)
                    final_balances.append(final_balance)

                # Bar graph containing final balances for each strategy
                cmap = get_cmap('tab20')
                bar_colors = [cmap(i % 20) for i in range(len(strategy_names))]
                plt.bar(strategy_names, final_balances, color=bar_colors)
                plt.xlabel('Strategy')
                plt.ylabel('Final Account Balance')
                plt.title(f'Final Account Balances For {currency_pair} {time_frame}')
                plt.savefig(f'../experiments/plots/{pair_time_str}_final_balances', bbox_inches='tight')
                plt.clf()

                # Export the final balances as a csv
                df = pd.DataFrame([final_balances], columns=strategy_names)
                df.to_csv(f'../experiments/results/final_balances_csv/{pair_time_str}_final_balances.csv')

    # Create plots of account value over time
    _create_account_value_plots()

    # Create bar graphs of the final balances
    _create_final_balance_bar_graphs()


if __name__ == "__main__":
    create_plots()
