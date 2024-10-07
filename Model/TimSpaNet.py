from Layers.embed import DataEmbedding
from Layers.Tem_extract import AutoCorrelationLayer,AutoCorrelation
from Layers.Spa_extract import Space
import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, units, time_step,layer, leaky_rate=0.2, n_heads = 2):
        super(Model, self).__init__()
        self.layer = layer
        self.unit = units
        self.time_step = time_step
        self.alpha = leaky_rate
        self.n_heads = n_heads
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.tem =  AutoCorrelationLayer(AutoCorrelation(),d_model= self.unit, n_heads=self.n_heads)
        self.spa =  Space(units = self.unit, time_step =self.time_step)
        
        self.W = nn.Parameter(torch.empty(size=(self.unit, self.layer)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        kernels = []
        for i in range(self.layer):
            kernels.append(nn.Sequential(nn.Conv1d( self.n_heads*3,  self.n_heads*3, kernel_size=(2 * i + 1,2 * i + 1), padding=(i,i)),
                                            nn.BatchNorm1d(self.n_heads*3),
                                            nn.ReLU()
                                            ))
        self.kernels = nn.ModuleList(kernels)
            
        # self.fc1  = nn.Linear(self.units,1)
        # self.fc2  = nn.Linear(1152 ,1152)
    def forward(self, x):
        mul_L, attention = self.spa(x)
        mul_L = mul_L.unsqueeze(1)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        X = X.unsqueeze(1)
        GX = torch.matmul(mul_L, X)
        batch_size, k, input_channel, node_cnt, time_step = GX.size()
        GX = GX.view(batch_size, -1, node_cnt, time_step)
        u=0
        for i in range(node_cnt):
            inp = GX[:,:,i,:]
            inp =  inp.permute(0,2,1)
            out , attn = self.tem(inp,inp,inp)
            B,T,H,E  = out.shape
            out = out.view(B,T,-1).permute(0,2,1)
            globals()['x%s' % i]=self.kernels[i](out)
            print(globals()['x%s' % i].shape)
            u+=globals()['x%s' % i]
        print(u.shape)
            

        z = self.bn(u.mean(3).mean(-1))
        z = F.relu(z)
        z = torch.matmul(z,self.W)
        c = F.softmax(z, dim=1)
        #shape(B,num_layer)
        v=0
        for i in range(self.layer):
            v+= ((globals()['x%s' % i])*(c[:,i].view(-1,1,1,1)))
        out = v+x
        return out


        return  period_weight
 