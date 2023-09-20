import numpy as np
import pandas as pd
from typing import Tuple


class Model:
    def __init__(self, name: str) -> None:
        self.name = name

    def train(self, df: pd.DataFrame) -> None:
        pass

    def load_model(self) -> None:
        pass

    def predict(self, x: np.array) -> Tuple[float, float, float, float]:
        pass
