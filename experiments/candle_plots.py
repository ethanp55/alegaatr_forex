from data.data_loader import DataLoader
import matplotlib.pyplot as plt
import mplfinance as mpf
from utils.utils import CURRENCY_PAIRS, YEARS


def candle_plots() -> None:
    for currency_pair in CURRENCY_PAIRS:
        for year in YEARS[1:]:
            year_data = DataLoader.load_simulation_data(currency_pair, 'H4', False, year)

            year_data.set_index('Date', inplace=True)
            year_data = year_data[['Mid_Open', 'Mid_High', 'Mid_Low', 'Mid_Close']]
            year_data.columns = ['Open', 'High', 'Low', 'Close']

            mpf.plot(year_data,
                     type='candle',
                     title=f'{currency_pair} H4 {year}',
                     style='yahoo',
                     volume=False,
                     figratio=(12.00, 5.75),
                     returnfig=True,
                     show_nontrading=False,
                     )

            plt.savefig(f'../experiments/plots/candle_plots/{currency_pair}_H4_{year}', bbox_inches='tight')
            plt.clf()


def overlapping_line_plots() -> None:
    for currency_pair in CURRENCY_PAIRS:
        for year in YEARS[2:]:
            year_data = DataLoader.load_simulation_data(currency_pair, 'H4', False, year)
            prev_year_data = DataLoader.load_simulation_data(currency_pair, 'H4', False, year - 1)

            def _keep_time_part(date):
                return date.replace(year=2000)

            # Create new columns with only month, day, and hour from 'Date'
            year_data['Date_join'] = year_data['Date'].apply(_keep_time_part)
            prev_year_data['Date_join'] = prev_year_data['Date'].apply(_keep_time_part)

            # Perform inner join on the 'Date_join' column
            joined_data = year_data.merge(prev_year_data, on='Date_join', how='inner')

            # Plot
            year_closes = joined_data['Mid_Close_x']
            prev_year_closes = joined_data['Mid_Close_y']
            plt.figure(figsize=(5, 3))
            plt.plot(prev_year_closes, label=f'{year - 1} Closes', color='blue')
            plt.plot(year_closes, label=f'{year} Closes', color='orange')
            plt.xlabel('Time Step', fontsize=22, fontweight='bold')
            plt.ylabel('Price', fontsize=22, fontweight='bold')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(loc='upper right')
            plt.savefig(f'../experiments/plots/candle_plots/{currency_pair}_H4_{year - 1}_vs_{year}',
                        bbox_inches='tight')
            plt.clf()


if __name__ == '__main__':
    # candle_plots()

    overlapping_line_plots()
