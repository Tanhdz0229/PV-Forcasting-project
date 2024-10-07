
from utils.tools import EarlyStopping
from Model.TimSpaNet import Model
from utils.metrics import metric
from Experiments.ex_base import Exp_Basic
from Data.DataLoader import data_provider

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import numpy as np


class Exp_Main(Exp_Basic):
    def __init__(self, learning_rate,patience,train_epochs, units, time_step):
        super(Exp_Main, self).__init__(learning_rate, patience, train_epochs, units, time_step)
        self.learning_rate = learning_rate
        self.patience = patience
        self.train_epochs = train_epochs
        self.units = units
        self.time_step = time_step

    def _build_model(self):
        model = Model(units = 3, time_step = 144*7,layer= 4, leaky_rate=0.2,n_heads = 2).float()
        return model

    def _get_data(self):
        data_loader = data_provider()
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, x):
        def _run_model():
            outputs = self.model(x)
            return outputs
        outputs = _run_model()
        return outputs
    # def vali(self, vali_loader, criterion):
    #     total_loss = []
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

    #             pred = outputs.detach()
    #             true = batch_y.detach()

    #             loss = criterion(pred, true)

    #             total_loss.append(loss)
    #     total_loss = np.average(total_loss)
    #     self.model.train()
    #     return total_loss

    def train(self):
        train_loader = self._get_data()
        # vali_loader = self._get_data()
        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        # early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, tim) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                outputs = self._predict(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_loader, criterion)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
            #     epoch + 1, train_steps, train_loss, vali_loss))
            # early_stopping(z, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
        #     #     break
        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        return self.model.eval()