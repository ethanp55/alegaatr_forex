import matplotlib.pyplot as plt
import pickle
from utils.utils import CURRENCY_PAIRS, TIME_FRAMES


def process_alegaatr_metrics() -> None:
    def process_prediction_values() -> None:
        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                file_path = f'../experiments/results/alegaatr_metrics/{currency_pair}_{time_frame}'
                predictions_when_wrong = pickle.load(open(f'{file_path}_predictions_when_wrong.pickle', 'rb'))
                trade_values_when_wrong = pickle.load(open(f'{file_path}_trade_values_when_wrong.pickle', 'rb'))
                predictions_when_correct = pickle.load(open(f'{file_path}_predictions_when_correct.pickle', 'rb'))
                trade_values_when_correct = pickle.load(open(f'{file_path}_trade_values_when_correct.pickle', 'rb'))

                assert len(predictions_when_wrong) == len(trade_values_when_wrong) and len(
                    predictions_when_correct) == len(trade_values_when_correct)

                n_bins = int(0.25 * len(predictions_when_wrong))

                plt.grid()
                plt.hist(predictions_when_wrong, bins=n_bins, alpha=0.75, label='Predictions', color='red')
                plt.hist(trade_values_when_wrong, bins=n_bins, alpha=0.5, label='Trade Amounts', color='green')
                plt.xlabel('USD Amounts')
                plt.ylabel('Counts')
                plt.legend(loc='best')
                plt.title(f'Distributions of Incorrect Predictions and Trade Amounts on {currency_pair} {time_frame}')
                plt.savefig(f'../experiments/plots/alegaatr_metrics/{currency_pair}_{time_frame}_incorrect',
                            bbox_inches='tight')
                plt.clf()

                n_bins = int(0.25 * len(predictions_when_correct))

                plt.grid()
                plt.hist(predictions_when_correct, bins=n_bins, alpha=0.75, label='Predictions', color='blue')
                plt.hist(trade_values_when_correct, bins=n_bins, alpha=0.5, label='Trade Amounts', color='green')
                plt.xlabel('USD Amounts')
                plt.ylabel('Counts')
                plt.legend(loc='best')
                plt.title(f'Distributions of Correct Predictions and Trade Amounts on {currency_pair} {time_frame}')
                plt.savefig(f'../experiments/plots/alegaatr_metrics/{currency_pair}_{time_frame}_correct',
                            bbox_inches='tight')
                plt.clf()

    # Process AlegAATr's prediction values to see what they look when they are correct and incorrect
    process_prediction_values()


if __name__ == "__main__":
    process_alegaatr_metrics()
