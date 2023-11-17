import pandas as pd
from typing import Optional
from utils.utils import TEST_START_DATE, YEARS


class DataLoader(object):
    @staticmethod
    def load_simulation_data(currency_pair: str, time_frame: str, optimize: bool,
                             year: Optional[int] = None) -> pd.DataFrame:
        df = pd.read_csv(f'../data/files/Oanda_{currency_pair}_{time_frame}_2013-2023.csv')
        df.Date = pd.to_datetime(df.Date, utc=True)

        if optimize:
            df = df.loc[df['Date'] < TEST_START_DATE]

        else:
            assert year is not None and year in YEARS
            start_date = f'{year}-11-01 00:00:00'
            end_date = f'{year + 1}-11-01 00:00:00'
            df = df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date)]

        df.reset_index(drop=True, inplace=True)

        return df
