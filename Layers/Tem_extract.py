
import torch
import torch.nn as nn
import math

class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor           = factor
        self.scale            = scale
        self.mask_flag        = mask_flag
        self.output_attention = output_attention
        self.dropout          = nn.Dropout(attention_dropout)
    def time_delay_agg_training(self, values, corr):
        head       = values.shape[1]
        channel    = values.shape[2]
        length     = values.shape[3]
        # find top k
        top_k      = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index      = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights    = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr   = torch.softmax(weights, dim=-1)

class AutoCorrelation(nn.Module):
    def __init__(self, factor=1, scale=None, attention_dropout=0.1, output_attention=False,pre_leg=144):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.pre_leg = pre_leg
        self.dropout = nn.Dropout(attention_dropout)
    def temporal_corr(self, values, corr):
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
>>>>>>> df411a8b574a4e5f82f2df58d0298d93cccb0bb1
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
<<<<<<< HEAD
            pattern    = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros  = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys   = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys   = keys[:, :L, :, :]
        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res   = q_fft * torch.conj(k_fft)
        corr  = torch.fft.irfft(res, n=L, dim=-1)
        V     = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys   = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
        self.query_projection  = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection    = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection  = nn.Linear(d_model, d_values * n_heads)
        self.out_projection    = nn.Linear(d_values * n_heads, d_model)
        self.n_heads           = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _   = queries.shape
        _, S, _   = keys.shape
        H         = self.n_heads
        queries   = self.query_projection(queries).view(B, L, H, -1)
        keys      = self.key_projection(keys).view(B, S, H, -1)
        values    = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out       = out.view(B, L, -1)
        return self.out_projection(out), attn
=======
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        values = values[:, :L, :, :]
        keys = keys[:, :L, :, :]
        #  Wiener–Khinchin theorem
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)
        V = self.temporal_corr(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        return V.contiguous(), corr.permute(0, 3, 1, 2)



class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads=2, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        # d_keys = d_keys or (d_model // n_heads)
        # d_values = d_values or (d_model // n_heads)
        d_keys = 3
        d_values = 3

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values
        )


        return out, attn
=======
>>>>>>> f597dbd8bd71228ea1b259f3c4d32bcd932b7e37
>>>>>>> df411a8b574a4e5f82f2df58d0298d93cccb0bb1
