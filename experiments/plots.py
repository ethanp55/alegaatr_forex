import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def create_plots() -> None:
    def _create_account_value_plots() -> None:
        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[1:]:
                    pair_time_year_str = f'{currency_pair}_{time_frame}_{year}'
                    file_list = os.listdir('../experiments/results/')
                    filtered_file_list = [file_name for file_name in file_list if
                                          (pair_time_year_str in file_name and 'account_values' in file_name)]

                    account_values, strategy_names, max_len = [], [], 0

                    for file_name in filtered_file_list:
                        strategy_name = file_name.split('_')[0]
                        values = pickle.load(open(f'../experiments/results/{file_name}', 'rb'))

                        strategy_names.append(strategy_name)
                        account_values.append(values)
                        max_len = max(max_len, len(values))

                        # Plot for each strategy
                        x = range(0, len(values))
                        plt.grid()
                        plt.plot(x, values)
                        plt.xlabel('Time')
                        plt.ylabel('Account Balance')
                        plt.title(f'Account Balances Over Time For {strategy_name} on {pair_time_year_str}')
                        plt.savefig(f'../experiments/plots/{strategy_name}_{pair_time_year_str}_account_values',
                                    bbox_inches='tight')
                        plt.clf()

                    # Plot containing all strategies
                    x = list(range(0, max_len))
                    x_common = np.array(x)
                    plt.figure(figsize=(10, 6))
                    plt.grid()

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
                    plt.title(f'Account Balances Over Time on {pair_time_year_str}')
                    legend = plt.legend(loc='best')
                    for text in legend.get_texts():
                        text.set_fontsize(8)
                    plt.savefig(f'../experiments/plots/{pair_time_year_str}_account_values', bbox_inches='tight')
                    plt.clf()

    def _create_final_balance_bar_graphs() -> None:
        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[1:]:
                    pair_time_year_str = f'{currency_pair}_{time_frame}_{year}'
                    file_list = os.listdir('../experiments/results/')
                    filtered_file_list = [file_name for file_name in file_list if
                                          (pair_time_year_str in file_name and 'final_balances' in file_name)]

                    final_balances, strategy_names, max_len = [], [], 0

                    for file_name in filtered_file_list:
                        strategy_name = file_name.split('_')[0]
                        final_balance = pickle.load(open(f'../experiments/results/{file_name}', 'rb'))

                        strategy_names.append(strategy_name)
                        final_balances.append(final_balance)

                    # Bar graph containing final balances for each strategy
                    names_to_colors = pickle.load(open('./plots/color_mappings.pickle', 'rb'))
                    names_to_colors['Random'] = 'cyan'
                    names_to_colors['Random2'] = 'cyan'
                    names_to_colors['Random05'] = 'cyan'
                    names_to_colors['Random01'] = 'cyan'
                    names_to_colors['Random001'] = 'cyan'
                    bar_colors = [names_to_colors[name] for name in strategy_names]
                    plt.grid()
                    plt.bar(strategy_names, final_balances, color=bar_colors)
                    plt.xlabel('Strategy')
                    plt.xticks(rotation=90)
                    plt.ylabel('Final Account Balance')
                    plt.title(f'Final Account Balances on {pair_time_year_str}')
                    plt.savefig(f'../experiments/plots/{pair_time_year_str}_final_balances', bbox_inches='tight')
                    plt.clf()

                    # Export the final balances as a csv
                    df = pd.DataFrame([final_balances], columns=strategy_names)
                    df.to_csv(f'../experiments/results/final_balances_csv/{pair_time_year_str}_final_balances.csv')

    def _corr_n_trades_profit() -> None:
        # Get the number of trades for each strategy
        strategies_n_trades = {}

        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[1:]:
                    pair_time_year_str = f'{currency_pair}_{time_frame}_{year}'
                    file_list = os.listdir('../experiments/results/')
                    filtered_file_list = [file_name for file_name in file_list if
                                          (pair_time_year_str in file_name and 'trade_amounts' in file_name)]

                    for file_name in filtered_file_list:
                        strategy_name = file_name.split('_')[0]
                        amounts = pickle.load(open(f'../experiments/results/{file_name}', 'rb'))

                        strategies_n_trades[strategy_name] = strategies_n_trades.get(strategy_name, 0) + len(amounts)

        strategies_profits = {}
        directory = '../experiments/results/final_balances_csv/'
        file_list = os.listdir(directory)

        for file_name in file_list:
            df = pd.read_csv(f'{directory}{file_name}')

            for strategy in df.columns[1:]:
                profit = df.loc[df.index[0], strategy] - 10000
                strategies_profits[strategy] = strategies_profits.get(strategy, []) + [profit]

        x_vals = [val for val in strategies_n_trades.values()]
        y_vals = [sum(strategies_profits[key]) / len(strategies_profits[key]) for key in strategies_n_trades.keys()]
        # Calculate the Pearson correlation coefficient
        correlation, p_value = pearsonr(x_vals, y_vals)

        name_to_class = {
            'AlegAATr': 'Bandit',
            'Choc': 'Trained Rule',
            'RSI': 'Trained Rule',
            'UCB': 'Bandit',
            'EXP3': 'Bandit',
            'MACD': 'Trained Rule',
            'SqueezePro': 'Trained Rule',
            'MACDKeyLevel': 'Trained Rule',
            'EEE': 'Bandit',
            'MACrossover': 'Trained Rule',
            'MACDStochastic': 'Trained Rule',
            'Supertrend': 'Trained Rule',
            'Ensemble': 'Ensemble',
            'BollingerBands': 'Trained Rule',
            'LstmMixture': 'Price Forecaster',
            'Lstm': 'Price Forecaster',
            'KeltnerChannels': 'Trained Rule',
            'BarMovement': 'Trained Rule',
            'Stochastic': 'Trained Rule',
            'KNN': 'Price Forecaster',
            'CNN': 'Price Forecaster',
            'MLP': 'Price Forecaster',
            'RandomForest': 'Price Forecaster',
            'PSAR': 'Trained Rule',
            'BeepBoop': 'Trained Rule',
            'Random': 'Random',
            'Random2': 'Random',
            'Random05': 'Random',
            'Random01': 'Random',
            'Random001': 'Random'
        }

        class_to_color = {
            'Bandit': 'blue',
            'Trained Rule': 'red',
            'Ensemble': 'green',
            'Price Forecaster': 'purple',
            'Random': 'orange'
        }

        # Add text annotation with correlation coefficient (rounded to 2 decimals)
        plt.figure(figsize=(9, 3))
        plt.text(0.175, 0.175, f"r = {correlation:.2f}", ha='center', transform=plt.gca().transAxes,
                 fontsize=16, fontweight='bold')
        plt.grid()

        for key in strategies_n_trades.keys():
            n_trades, avg_amount = strategies_n_trades[key], sum(strategies_profits[key]) / len(strategies_profits[key])
            label = name_to_class[key]
            color = class_to_color[label]
            plt.scatter(n_trades, avg_amount, c=color)

        # Manual legend
        for label, color in class_to_color.items():
            plt.scatter([], [], c=color, label=label)

        plt.legend(loc='best')
        plt.text(strategies_n_trades['Random'] - 100,
                 sum(strategies_profits['Random']) / len(strategies_profits['Random']),
                 'Random - 0.33', fontsize=10, ha='right')
        plt.text(strategies_n_trades['Random2'] - 100,
                 sum(strategies_profits['Random2']) / len(strategies_profits['Random2']),
                 'Random - 0.2', fontsize=10, ha='right')
        plt.text(strategies_n_trades['Random05'] - 100,
                 sum(strategies_profits['Random05']) / len(strategies_profits['Random05']),
                 'Random - 0.05', fontsize=10, ha='right')
        plt.text(strategies_n_trades['Random01'] - 400,
                 sum(strategies_profits['Random01']) / len(strategies_profits['Random01']),
                 'Random - 0.01', fontsize=10, ha='right')
        plt.text(strategies_n_trades['Random001'] + 3000,
                 sum(strategies_profits['Random001']) / len(strategies_profits['Random001']),
                 'Random - 0.001', fontsize=10, ha='right')
        plt.xlabel('Number of Trades', fontsize=22, fontweight='bold')
        plt.ylabel('Amount', fontsize=22, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(f'../experiments/plots/report/n_trades_profit_corr', bbox_inches='tight')
        plt.clf()

    # Create plots of account value over time
    # _create_account_value_plots()

    # Create bar graphs of the final balances
    # _create_final_balance_bar_graphs()

    # Create correlation graph between number of trades and average profit amount
    _corr_n_trades_profit()


if __name__ == "__main__":
    create_plots()
