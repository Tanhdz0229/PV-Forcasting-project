import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch

from torch.utils.data import DataLoader

class Dataset_Custom(Dataset):
    def __init__(self, root_path, scale  = True):
        # size [seq_len, label_len, pred_len]
        self.seq_len = 144*7
        self.pred_len = 144
        self.root_path = root_path
        self.scale = scale
        self.__read_data__()
        self.x_end_idx = self.get_x_end_idx()
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.root_path)
        df_raw['Measurement Time'] = pd.to_datetime(df_raw['Measurement Time'])
        df_raw['Date'] = df_raw['Measurement Time'].dt.day
        df_raw['Month'] = df_raw['Measurement Time'].dt.month
        df_raw['Year'] = df_raw['Measurement Time'].dt.year
        df_raw['Hour'] = df_raw['Measurement Time'].dt.hour
        df_raw['Minute'] = df_raw['Measurement Time'].dt.minute
        df_time = df_raw[['Year','Month','Date','Hour','Minute']].values
        df_values = df_raw[[' P1 Solar Irradiance',' P1 Temperature',' P1 Total AC Power (Positive)']]
        if self.scale:
            data = df_values
            self.scaler.fit(data.values)
            data = self.scaler.transform(data.values)
        else:
            data = df_values.values
        self.data_values = df_values
        self.data_time = df_time
        self.df_leng = len(self.data_values)

    def __getitem__(self, index):
        label_end   =  self.x_end_idx[index]
        seq_start   =  label_end - self.pred_len - self.seq_len
        seq_end     =  label_end - self.pred_len
        train_data  =  self.data_values.iloc[seq_start:seq_end].values
        target_data =  self.data_values.iloc[seq_start:label_end, 2].values
        train_time =  self.data_time[seq_start:label_end]
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        t = torch.from_numpy(train_time).type(torch.float)
        return x, y, t
    def __len__(self):
        return  len(self.x_end_idx)
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    def get_x_end_idx(self):
        x_index_set = range(self.seq_len+self.pred_len, self.df_leng +self.pred_len)
        x_end_idx = [x_index_set[j* self.pred_len] for j in range((len(x_index_set)) // self.pred_len)]
        return x_end_idx


def data_provider():
    data_set = Dataset_Custom(root_path= 'C:\\Thanhnt\\PV-Forcasting-project\\Data\\Dataset\\Shibaura\\2012_2017.csv',  scale  = True
    )
    data_loader = DataLoader(
        data_set,
        batch_size=32 ,
        shuffle  =True)
    return data_loader
