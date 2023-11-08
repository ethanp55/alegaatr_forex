from aat.assumptions import Assumptions
from market_proxy.trade import Trade
import numpy as np
from pandas import DataFrame
import pickle
from sklearn.neighbors import NearestNeighbors


class AATTrainer:
    def __init__(self, currency_pair: str, strategy_name: str, time_frame: str, year: int) -> None:
        self.currency_pair, self.strategy_name, self.time_frame, self.year = \
            currency_pair, strategy_name, time_frame, year
        self.recent_tuple, self.training_data = None, []

    # Adds a new AAT tuple to its training data
    def create_new_tuple(self, df: DataFrame, curr_idx: int, trade: Trade) -> None:
        assert self.recent_tuple is None

        trade_amount = abs(trade.open_price - trade.stop_loss) * trade.n_units
        new_assumptions = Assumptions(df, curr_idx, self.currency_pair, trade_amount)

        self.recent_tuple = new_assumptions.create_aat_tuple()

    # Adds the AAT correction term (what AAT is trying to learn) to the newest AAT tuple, stores the tuple, then resets
    # the tuple
    def add_correction_term(self, final_trade_amount: float) -> None:
        assert self.recent_tuple is not None

        predicted_amount = self.recent_tuple[-1]
        correction_term = final_trade_amount / predicted_amount

        self.recent_tuple.append(correction_term)
        self.training_data.append(self.recent_tuple)

        # Reset the tuple for the next training iteration
        self.recent_tuple = None

    # Trains a KNN model, saves the model, and saves the AAT correction terms
    def save(self) -> None:
        name_pair_time_year_str = f'{self.strategy_name}_{self.currency_pair}_{self.time_frame}_{self.year}'

        x = np.array(self.training_data, dtype=float)[:, 0:-2]  # Assumptions - convert any booleans to floats
        y = np.array(self.training_data)[:, -1]  # Correction terms
        n_neighbors = int(len(x) ** 0.5)

        print(f'X and Y data for {name_pair_time_year_str}')
        print('X train shape: ' + str(x.shape))
        print('Y train shape: ' + str(y.shape))
        print(f'N neighbors: {n_neighbors}')

        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(x)

        correction_terms_file_name = f'../aat/training_data/{name_pair_time_year_str}_aat_correction_terms.pickle'
        knn_file_name = f'../aat/training_data/{name_pair_time_year_str}_aat_knn.pickle'

        # Save the corrections and KNN model
        with open(correction_terms_file_name, 'wb') as f:
            pickle.dump(y, f)

        with open(knn_file_name, 'wb') as f:
            pickle.dump(knn, f)
