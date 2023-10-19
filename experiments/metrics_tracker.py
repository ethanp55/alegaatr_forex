import pickle
from typing import List


class MetricsTracker:
    # For each strategy, want to keep track of:
    #   - Trade amounts (array of trade amounts)
    #       - With these, can calculate win rate and average trade amount
    #   - Account value over time (array of account values)
    #   - Final balance
    #   - Number of scenarios where profitable in testing / number of scenarios where profitable in training
    def __init__(self) -> None:
        self.trade_amounts, self.account_values, self.final_balances = {}, {}, {}
        self.profitable_training, self.profitable_testing, self.profitable_ratios = {}, {}, {}

    def increment_profitable_training(self, strategy_name: str, currency_pair: str, time_frame: str,
                                      profitable: bool) -> None:
        strategy_pair_time_str = f'{strategy_name}_{currency_pair}_{time_frame}'
        amount = 1 if profitable else 0
        self.profitable_training[strategy_pair_time_str] = amount

    def increment_profitable_testing(self, strategy_name: str, currency_pair: str, time_frame: str,
                                     profitable: bool) -> None:
        strategy_pair_time_str = f'{strategy_name}_{currency_pair}_{time_frame}'
        amount = 1 if profitable else 0
        self.profitable_testing[strategy_pair_time_str] = amount

    def calculate_profitable_ratios(self, strategy_names: List[str]) -> None:
        for strategy_name in strategy_names:
            numerator, denominator = 0, 0

            for key in self.profitable_training:
                if strategy_name in key:
                    numerator += self.profitable_testing[key]
                    denominator += self.profitable_training[key]

            self.profitable_ratios[strategy_name] = (numerator / denominator) if denominator > 0 else None

    def update_trade_amounts(self, strategy_name: str, currency_pair: str, time_frame: str, trade_amount: float,
                             account_balance: float) -> None:
        strategy_pair_time_str = f'{strategy_name}_{currency_pair}_{time_frame}'

        self.trade_amounts[strategy_pair_time_str] = self.trade_amounts.get(strategy_pair_time_str, []) + [
            trade_amount]

        # We can also update the account values here because account value changes with each trade
        self.account_values[strategy_pair_time_str] = self.account_values.get(strategy_pair_time_str, []) + [
            account_balance]

    def update_final_balance(self, strategy_name: str, currency_pair: str, time_frame: str,
                             final_balance: float) -> None:
        strategy_pair_time_str = f'{strategy_name}_{currency_pair}_{time_frame}'

        self.final_balances[strategy_pair_time_str] = final_balance

    def save_data(self, strategy_names: List[str]) -> None:
        # Save the trade amounts
        for key, val in self.trade_amounts.items():
            file_location = f'../experiments/results/{key}_trade_amounts.pickle'

            with open(file_location, 'wb') as f:
                pickle.dump(val, f)

        # Save the account values over time
        for key, val in self.account_values.items():
            file_location = f'../experiments/results/{key}_account_values.pickle'

            with open(file_location, 'wb') as f:
                pickle.dump(val, f)

        # Save the final balances
        for key, val in self.final_balances.items():
            file_location = f'../experiments/results/{key}_final_balances.pickle'

            with open(file_location, 'wb') as f:
                pickle.dump(val, f)

        # Save the profitable testing set ratios
        self.calculate_profitable_ratios(strategy_names)

        for key, val in self.profitable_ratios.items():
            file_location = f'../experiments/results/{key}_profitable_ratios.pickle'

            with open(file_location, 'wb') as f:
                pickle.dump(val, f)


