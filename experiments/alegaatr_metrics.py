import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def process_alegaatr_metrics() -> None:
    def process_all_prediction_values() -> None:
        predictions_when_wrong, trade_values_when_wrong, predictions_when_correct, trade_values_when_correct = \
            [], [], [], []

        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[2:]:
                    file_path = f'../experiments/results/alegaatr_metrics/{currency_pair}_{time_frame}_{year}'
                    predictions_when_wrong += pickle.load(open(f'{file_path}_predictions_when_wrong.pickle', 'rb'))
                    trade_values_when_wrong += pickle.load(open(f'{file_path}_trade_values_when_wrong.pickle', 'rb'))
                    predictions_when_correct += pickle.load(open(f'{file_path}_predictions_when_correct.pickle', 'rb'))
                    trade_values_when_correct += pickle.load(
                        open(f'{file_path}_trade_values_when_correct.pickle', 'rb'))

        predictions_when_wrong_clean = [val for val in predictions_when_wrong if val != np.inf]
        predictions_when_correct_clean = [val for val in predictions_when_correct if val != np.inf]

        n_bins = int(0.25 * len(predictions_when_wrong_clean))

        plt.grid()
        plt.hist(predictions_when_wrong_clean, bins=n_bins, alpha=0.75, label='Predictions', color='red')
        plt.hist(trade_values_when_wrong, bins=n_bins, alpha=0.5, label='Trade Amounts', color='green')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Incorrect Predictions and Trade Amounts')
        plt.savefig(f'../experiments/plots/report/incorrect', bbox_inches='tight')
        plt.clf()

        n_bins = int(0.25 * len(predictions_when_correct_clean))

        plt.grid()
        plt.hist(predictions_when_correct_clean, bins=n_bins, alpha=0.75, label='Predictions', color='blue')
        plt.hist(trade_values_when_correct, bins=n_bins, alpha=0.5, label='Trade Amounts', color='green')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Correct Predictions and Trade Amounts')
        plt.savefig(f'../experiments/plots/report/correct', bbox_inches='tight')
        plt.clf()

        plt.grid()
        plt.hist(predictions_when_correct_clean, bins=n_bins, alpha=0.75, label='Correct Predictions', color='green')
        plt.hist(predictions_when_wrong_clean, bins=n_bins, alpha=0.5, label='Incorrect Predictions', color='red')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Correct and Incorrect Predictions')
        plt.savefig(f'../experiments/plots/report/correct_incorrect', bbox_inches='tight')
        plt.clf()

    def process_worst_and_best_values() -> None:
        predictions_when_wrong_best, predictions_when_wrong_worst, trade_values_when_wrong_best, \
        trade_values_when_wrong_worst, predictions_when_correct_best, predictions_when_correct_worst, \
        trade_values_when_correct_best, trade_values_when_correct_worst = [], [], [], [], [], [], [], []
        best_pair_time_str, worst_pair_time_str = 'Usd_Jpy_M30', 'Gbp_Chf_H1'

        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[2:]:
                    file_path = f'../experiments/results/alegaatr_metrics/{currency_pair}_{time_frame}_{year}'

                    if best_pair_time_str in file_path:
                        predictions_when_wrong_best += pickle.load(
                            open(f'{file_path}_predictions_when_wrong.pickle', 'rb'))
                        trade_values_when_wrong_best += pickle.load(
                            open(f'{file_path}_trade_values_when_wrong.pickle', 'rb'))
                        predictions_when_correct_best += pickle.load(
                            open(f'{file_path}_predictions_when_correct.pickle', 'rb'))
                        trade_values_when_correct_best += pickle.load(
                            open(f'{file_path}_trade_values_when_correct.pickle', 'rb'))

                    elif worst_pair_time_str in file_path:
                        predictions_when_wrong_worst += pickle.load(
                            open(f'{file_path}_predictions_when_wrong.pickle', 'rb'))
                        trade_values_when_wrong_worst += pickle.load(
                            open(f'{file_path}_trade_values_when_wrong.pickle', 'rb'))
                        predictions_when_correct_worst += pickle.load(
                            open(f'{file_path}_predictions_when_correct.pickle', 'rb'))
                        trade_values_when_correct_worst += pickle.load(
                            open(f'{file_path}_trade_values_when_correct.pickle', 'rb'))

        predictions_when_wrong_best_clean = [val for val in predictions_when_wrong_best if val != np.inf]
        predictions_when_correct_best_clean = [val for val in predictions_when_correct_best if val != np.inf]
        predictions_when_wrong_worst_clean = [val for val in predictions_when_wrong_worst if val != np.inf]
        predictions_when_correct_worst_clean = [val for val in predictions_when_correct_worst if val != np.inf]

        n_bins = int(0.25 * len(predictions_when_wrong_best_clean))

        plt.grid()
        plt.hist(predictions_when_wrong_best_clean, bins=n_bins, alpha=0.75, label='Predictions', color='red')
        plt.hist(trade_values_when_wrong_best, bins=n_bins, alpha=0.5, label='Trade Amounts', color='green')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Incorrect Predictions and Trade Amounts on USD/JPY M30')
        plt.savefig(f'../experiments/plots/report/incorrect_usd_jpy_m30', bbox_inches='tight')
        plt.clf()

        n_bins = int(0.25 * len(predictions_when_correct_best_clean))

        plt.grid()
        plt.hist(predictions_when_correct_best_clean, bins=n_bins, alpha=0.75, label='Predictions', color='blue')
        plt.hist(trade_values_when_correct_best, bins=n_bins, alpha=0.5, label='Trade Amounts', color='green')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Correct Predictions and Trade Amounts on USD/JPY M30')
        plt.savefig(f'../experiments/plots/report/correct_usd_jpy_m30', bbox_inches='tight')
        plt.clf()

        n_bins = int(0.25 * len(predictions_when_wrong_worst_clean))

        plt.grid()
        plt.hist(predictions_when_wrong_worst_clean, bins=n_bins, alpha=0.75, label='Predictions', color='red')
        plt.hist(trade_values_when_wrong_worst, bins=n_bins, alpha=0.5, label='Trade Amounts', color='green')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Incorrect Predictions and Trade Amounts on GBP/CHF H1')
        plt.savefig(f'../experiments/plots/report/incorrect_gbp_chf_h1', bbox_inches='tight')
        plt.clf()

        plt.grid()
        plt.hist(predictions_when_correct_worst_clean, alpha=0.75, label='Predictions', color='blue')
        plt.hist(trade_values_when_correct_worst, alpha=0.5, label='Trade Amounts', color='green')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Correct Predictions and Trade Amounts on GBP/CHF H1')
        plt.savefig(f'../experiments/plots/report/correct_gbp_chf_h1', bbox_inches='tight')
        plt.clf()

        n_bins = int(0.25 * len(predictions_when_correct_best_clean))

        plt.grid()
        plt.hist(predictions_when_correct_best_clean, bins=n_bins, alpha=0.75, label='Correct Predictions',
                 color='green')
        plt.hist(predictions_when_wrong_best_clean, bins=n_bins, alpha=0.5, label='Incorrect Predictions', color='red')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Correct and Incorrect Predictions on USD/JPY M30')
        plt.savefig(f'../experiments/plots/report/correct_incorrect_usd_jpy_m30', bbox_inches='tight')
        plt.clf()

        plt.grid()
        plt.hist(predictions_when_correct_worst_clean, alpha=0.75, label='Correct Predictions',
                 color='green')
        plt.hist(predictions_when_wrong_worst_clean, alpha=0.5, label='Incorrect Predictions', color='red')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Correct and Incorrect Predictions on GBP/CHF H1')
        plt.savefig(f'../experiments/plots/report/correct_incorrect_gbp_chf_h1', bbox_inches='tight')
        plt.clf()

        plt.grid()
        plt.hist(predictions_when_correct_best_clean, alpha=0.75, label='USD/JPY M30 Predictions',
                 color='green')
        plt.hist(predictions_when_correct_worst_clean, alpha=0.5, label='GBP/CHF H1 Predictions',
                 color='red')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Correct Predictions For USD/JPY M30 and GBP/CHF H1')
        plt.savefig(f'../experiments/plots/report/correct_usd_jpy_m30_gbp_chf_h1', bbox_inches='tight')
        plt.clf()

        n_bins = int(0.25 * len(predictions_when_wrong_best_clean))

        plt.grid()
        plt.hist(predictions_when_wrong_best_clean, bins=n_bins, alpha=0.75, label='USD/JPY M30 Predictions',
                 color='green')
        plt.hist(predictions_when_wrong_worst_clean, bins=n_bins, alpha=0.5, label='GBP/CHF H1 Predictions',
                 color='red')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.title(f'Distributions of Incorrect Predictions For USD/JPY M30 and GBP/CHF H1')
        plt.savefig(f'../experiments/plots/report/incorrect_usd_jpy_m30_gbp_chf_h1', bbox_inches='tight')
        plt.clf()

    # Process AlegAATr's prediction values to see how they look when they are correct and incorrect
    process_all_prediction_values()

    # Process AlegAATr's worst and best prediction values to see what they look like
    process_worst_and_best_values()


if __name__ == "__main__":
    process_alegaatr_metrics()
