from data.data_loader import DataLoader
import matplotlib.pyplot as plt
import mplfinance as mpf
from utils.utils import CURRENCY_PAIRS, YEARS

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
