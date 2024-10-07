from Experiments.Pipelines import Exp_Main

from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from torch.autograd import Variable

import numpy as np
import pandas as pd


model  = Exp_Main(learning_rate = 0.001 ,patience = 7,train_epochs = 200, units = 3, time_step = 144*7)
trained_model = model.train()

# lookBack=1008
# lookAhead=144
# test='C:\\Thanhnt\\PV-Forcasting-project\\Data\Dataset\\Shibaura\\2018_.csv'
# test1 = pd.read_csv(test)[[' P1 Solar Irradiance', ' P1 Temperature', ' P1 Total AC Power (Positive)']]
# time  = pd.read_csv(test)[['Measurement Time']]
# test1=test1.values
# scaler = MinMaxScaler(feature_range=(0, 1))
# test = scaler.fit_transform(test1)
# leng = len(test)
# x_index_set = range(lookBack+lookAhead, leng + lookAhead)
# x_end_idx = [x_index_set[j* lookAhead] for j in range((len(x_index_set)) // lookAhead)]

# pre = []
# act = []
# t = []
# for i in range(len(x_end_idx)):
#         label_end   =  x_end_idx[i]
#         seq_start   =  label_end - lookAhead - lookBack
#         seq_end     =  label_end - lookAhead

#         padding = torch.zeros([ label_end-seq_end, 3])
#         train_data = np.concatenate([test[seq_start: seq_end], padding])

#         train_data  =  torch.from_numpy(train_data).unsqueeze(0)
#         target_data =  test[seq_end : label_end,2]
#         target_data=target_data.reshape(len(target_data),1)
#         X = Variable(torch.Tensor(np.array(train_data)))
#         Y = Variable(torch.Tensor(np.array(target_data)))
#         time1 = time[seq_end : label_end]
#         predict = model(X)
#         predict = predict.data.numpy()
#         predict=predict[:,1008:1152]
#         Y=Y.cpu()
#         Y=Y.data.numpy()
#         pre.append(predict)
#         act.append(Y)
#         t.append(time1)
# pre2 = np.array(pre).reshape(-1,1)
# act2 = np.array(act).reshape(-1,1)
# data_predict=np.repeat(pre2,test1.shape[1],axis=-1)
# dataY_plot=np.repeat(act2,test1.shape[1],axis=-1)
# data_predict = scaler.inverse_transform(data_predict)[:,2]
# dataY_plot =scaler.inverse_transform(dataY_plot)[:,2]
# plt.axvline(x=len(Y), c='r', linestyle='--')
# from sklearn.metrics import r2_score

# plt.plot(data_predict,'b',label='Propose Model')
# plt.plot(dataY_plot,'r',label='Actual')

# plt.rcParams["figure.figsize"] = (400,3)
# plt.ylabel('PV power')
# plt.suptitle('Proposed')
# plt.legend(loc='best')
# plt.show()
# mse = mean_squared_error(dataY_plot, data_predict)
# rmse = mean_squared_error(dataY_plot, data_predict,squared=False)
# mae = mean_absolute_error(dataY_plot, data_predict)
# mape= mean_absolute_percentage_error(dataY_plot, data_predict)
# CC=r2_score(dataY_plot, data_predict)
# print('MSE==',mse,'RMSE==',rmse,'MAE==',mae,'MAPE==',mape,'CC==',CC)