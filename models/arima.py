import pickle
import pandas as pd
import pmdarima as pm
from models.model import Model
from utils.utils import ARIMA_START_DATE


class Arima(Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.arima_bid_pips_down, self.arima_bid_pips_up, self.arima_ask_pips_down, self.arima_ask_pips_up = \
            None, None, None, None

    def train(self, df: pd.DataFrame) -> None:
        # Fit an ARIMA model for each time series we're trying to predict
        print(f'Fitting ARIMA models for {self.name}...')

        df_train = df.loc[df['Date'] >= ARIMA_START_DATE]
        df_train.reset_index(drop=True, inplace=True)

        print(df_train.head())

        self.arima_bid_pips_down = pm.auto_arima(df_train['bid_pips_down'], stepwise=False, seasonal=False)
        self.arima_bid_pips_up = pm.auto_arima(df_train['bid_pips_up'], stepwise=False, seasonal=False)
        self.arima_ask_pips_down = pm.auto_arima(df_train['ask_pips_down'], stepwise=False, seasonal=False)
        self.arima_ask_pips_up = pm.auto_arima(df_train['ask_pips_up'], stepwise=False, seasonal=False)

        # Save the ARIMA info
        with open(f'./models/model_files/{self.name}_arima_bid_pips_down.pickle', 'wb') as f:
            pickle.dump(self.arima_bid_pips_down, f)

        with open(f'./models/model_files/{self.name}_arima_bid_pips_up.pickle', 'wb') as f:
            pickle.dump(self.arima_bid_pips_up, f)

        with open(f'./models/model_files/{self.name}_arima_ask_pips_down.pickle', 'wb') as f:
            pickle.dump(self.arima_ask_pips_down, f)

        with open(f'./models/model_files/{self.name}_arima_ask_pips_up.pickle', 'wb') as f:
            pickle.dump(self.arima_ask_pips_up, f)
