import pandas as pd


class Model:
    def __init__(self, name: str) -> None:
        self.name = name

    def train(self, df: pd.DataFrame) -> None:
        pass
