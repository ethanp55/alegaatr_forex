import pickle
import pandas as pd
import pmdarima as pm
from models.model import Model
from typing import Tuple
from utils.utils import ARIMA_START_DATE


class Arima(Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.arima_bid_pips_down, self.arima_bid_pips_up, self.arima_ask_pips_down, self.arima_ask_pips_up = \
            None, None, None, None

    def load_model(self) -> None:
        self.arima_bid_pips_down = pickle.load(
            open(f'../models/model_files/{self.name}_arima_bid_pips_down.pickle', 'rb'))
        self.arima_bid_pips_up = pickle.load(open(f'../models/model_files/{self.name}_arima_bid_pips_up.pickle', 'rb'))
        self.arima_ask_pips_down = pickle.load(
            open(f'../models/model_files/{self.name}_arima_ask_pips_down.pickle', 'rb'))
        self.arima_ask_pips_up = pickle.load(open(f'../models/model_files/{self.name}_arima_ask_pips_up.pickle', 'rb'))

    def predict(self, n_periods: int) -> Tuple[float, float, float, float]:
        arima_bid_pips_down_pred = self.arima_bid_pips_down.predict(n_periods=n_periods).iloc[-1,]
        arima_bid_pips_up_pred = self.arima_bid_pips_up.predict(n_periods=n_periods).iloc[-1,]
        arima_ask_pips_down_pred = self.arima_ask_pips_down.predict(n_periods=n_periods).iloc[-1,]
        arima_ask_pips_up_pred = self.arima_ask_pips_up.predict(n_periods=n_periods).iloc[-1,]

        return (arima_bid_pips_down_pred, arima_bid_pips_up_pred, arima_ask_pips_down_pred, arima_ask_pips_up_pred)

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
        with open(f'../models/model_files/{self.name}_arima_bid_pips_down.pickle', 'wb') as f:
            pickle.dump(self.arima_bid_pips_down, f)

        with open(f'../models/model_files/{self.name}_arima_bid_pips_up.pickle', 'wb') as f:
            pickle.dump(self.arima_bid_pips_up, f)

        with open(f'../models/model_files/{self.name}_arima_ask_pips_down.pickle', 'wb') as f:
            pickle.dump(self.arima_ask_pips_down, f)

        with open(f'../models/model_files/{self.name}_arima_ask_pips_up.pickle', 'wb') as f:
            pickle.dump(self.arima_ask_pips_up, f)
