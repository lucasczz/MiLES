import pandas as pd
import numpy as np
import lightning as L
from torch.utils import data
import pickle
import pathlib

DATAPATH = pathlib.Path(__file__).parent.parent.parent.joinpath("data", "processed")


class GeolifeModule(L.LightningDataModule):
    def __init__(self, n_hex_rows=50):
        self.n_hex_rows = n_hex_rows

    def prepare_data(self):
        with open(DATAPATH.joinpath("geolife_hex_{self.n_hex_rows}.pkl"), "rb") as f:
            self.df = pickle.loads(f)
        self.df = pd.get_dummies(self.df, columns=["user"])

    def setup(self, stage): ...

    def train_dataloader(self):
        return data.DataLoader(self.train)

    def val_dataloader(self):
        return data.DataLoader(self.val)

    def test_dataloader(self):
        return data.DataLoader(self.test)

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        ...

    def teardown(self):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        ...
