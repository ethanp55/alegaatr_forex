import pandas as pd
from utils.utils import OPTIMIZATION_START_DATE, TEST_START_DATE


class DataLoader(object):
    @staticmethod
    def load_training_data(file_path: str, pips_multiplier: int) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.Date = pd.to_datetime(df.Date, utc=True)
        df = df.loc[df['Date'] < OPTIMIZATION_START_DATE]
        df.reset_index(drop=True, inplace=True)

        # Create the labels (what we're trying to predict)
        df['bid_pips_down'] = abs(
            df['Bid_Open'] - df['Bid_Low']) * pips_multiplier
        df['bid_pips_up'] = abs(
            df['Bid_High'] - df['Bid_Open']) * pips_multiplier
        df['ask_pips_down'] = abs(
            df['Ask_Open'] - df['Ask_Low']) * pips_multiplier
        df['ask_pips_up'] = abs(
            df['Ask_High'] - df['Ask_Open']) * pips_multiplier

        return df

    @staticmethod
    def load_simulation_data(currency_pair: str, time_frame: str, optimize: bool,
                             year_range: str = '2021-2023') -> pd.DataFrame:
        df = pd.read_csv(f'../data/files/Oanda_{currency_pair}_{time_frame}_{year_range}.csv')
        df.Date = pd.to_datetime(df.Date, utc=True)
        df = df.loc[(df['Date'] >= OPTIMIZATION_START_DATE) & (df['Date'] < TEST_START_DATE)] if optimize else df.loc[
            df['Date'] >= TEST_START_DATE]
        df.reset_index(drop=True, inplace=True)

        return df
