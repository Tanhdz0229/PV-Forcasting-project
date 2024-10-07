import os
import torch


class Exp_Basic(object):
    def __init__(self,  learning_rate,patience,train_epochs, units, time_step):
        self.learning_rate = learning_rate
        self.patience = patience
        self.train_epochs = train_epochs
        self.units = units
        self.time_step = time_step
        self.model = self._build_model()
    def _build_model(self):
        raise NotImplementedError
        return None
    def _get_data(self):
        pass
    def vali(self):
        pass
    def train(self):
        pass
    def test(self):
        pass