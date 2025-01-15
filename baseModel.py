import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import HGTConv, Linear, BatchNorm
from torch_geometric.utils import scatter


# from GAT_with_batch import GAT


def cosine_func(x, y, epsilon=1e-8):
    x = x + epsilon
    y = y + epsilon
    return x * y / ((x ** 0.5) * (y ** 0.5))


def normalize_digraph(A):
    if len(A.shape) == 3:
        b, n, _ = A.shape
    elif len(A.shape) == 2:
        n, _ = A.shape
    node_degrees = A.detach().sum(dim=-1)
    degs_inv_sqrt = node_degrees ** -0.5
    degs_inv_sqrt[torch.isinf(degs_inv_sqrt)] = 0  # for text
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    if len(A.shape) == 3:
        norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
        norm_A = torch.bmm(torch.bmm(norm_degs_matrix, A), norm_degs_matrix)
    elif len(A.shape) == 2:
        norm_degs_matrix = norm_degs_matrix * degs_inv_sqrt.view(n, 1)
        norm_A = torch.mm(torch.mm(norm_degs_matrix, A), norm_degs_matrix)
    return norm_A


class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead=8, bias=True, dropout=0.1, norm_first=False):
        super().__init__()
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=True, batch_first=True)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _sa_block(self, x):
        x = self.self_attn(x, x, x,
                           attn_mask=None,
                           key_padding_mask=None,
                           need_weights=False, is_causal=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x):
        # if self.norm_first:
        #     x = x + self._sa_block(self.norm1(x))
        #     x = x + self._ff_block(self.norm2(x))
        # else:
        #     # x = self.norm1(x + self._sa_block(x))
        #     x = self.norm2(x + self._ff_block(x))

        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)

        # 计算位置编码值
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加可学习的权重
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        # 将位置编码与输入相加
        mask = torch.ones_like(x)
        mask[x.sum(dim=-1) == 0, :] = 0
        x = self.pe * mask
        return x

class GCN(nn.Module):
    def __init__(self, in_channels, neighbor_num, metric):
        super().__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels, self.in_channels)
        self.V = nn.Linear(self.in_channels, self.in_channels)
        self.bnv = nn.BatchNorm1d(self.in_channels)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1.)
        self.bnv.bias.data.zero_()

        self.ffn = FFN(d_model=self.in_channels, dim_feedforward=self.in_channels)

    def build_graph(self, x):
        if len(x.shape) == 3:
            b, n, c = x.shape
            if self.metric == 'dots':
                si = x.detach()
                si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
                threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
                adj = (si >= threshold).float()

            elif self.metric == 'cosine':
                si = x.detach()
                si = F.normalize(si, p=2, dim=-1)
                si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
                threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
                adj = (si >= threshold).float()

            elif self.metric == 'l1':
                si = x.detach().repeat(1, n, 1).view(b, n, n, c)
                si = torch.abs(si.transpose(1, 2) - si)
                si = si.sum(dim=-1)
                threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
                adj = (si <= threshold).float()

            elif self.metric == 'ones':
                adj = torch.ones(b, n, n, device=x.device)
            else:
                raise Exception("Error: wrong metric: ", self.metric)
            return adj
        elif len(x.shape) == 2:
            n, c = x.shape
            if self.metric == 'cosine':
                si = x.detach()
                si = F.normalize(si, p=2, dim=-1)
                si = torch.einsum('i j , j k -> i k', si, si.transpose(0, 1))
                threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, -1].view(n, 1)
                adj = (si >= threshold).float()

            else:
                raise Exception("Error: wrong metric: ", self.metric)
            return adj

    def forward(self, x):
        adj = self.build_graph(x.detach())
        A = normalize_digraph(adj)
        if len(x.shape) == 3:
            aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
            res = aggregate + self.U(x)
            res = self.bnv(res.transpose(1, 2)).transpose(1, 2)
            x = self.relu(x + res)
            x = self.ffn(x)
            # x = self.ffn(x, adj)
        # elif len(x.shape) == 2:
        #     aggregate = torch.einsum('i j, j k->i k', A, self.V(x))
        #     res = aggregate + self.U(x)
        #     res = self.bnv(res)
        #     x = self.relu(x + res).unsqueeze(0)
        #     x = self.ffn(x).squeeze(0)

        return x


class channelATT2_res(nn.Module):
    def __init__(self, dim, head=8):
        super().__init__()
        self.head = head
        self.atts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Tanh()) for _ in range(head)])

    def forward(self, x1, x2=None, x3=None, x4=None, x5=None):
        if x5 is not None:
            x = torch.cat([x1, x2, x3, x4, x5], dim=-1)
        elif x4 is not None:
            x = torch.cat([x1, x2, x3, x4], dim=-1)
        elif x3 is not None:
            x = torch.cat([x1, x2, x3], dim=-1)
        elif x2 is not None:
            x = torch.cat([x1, x2], dim=-1)
        else:
            x = x1
        out = []
        for i in range(self.head):
            att = self.atts[i](x)
            head_out = att * x + x
            out.append(head_out)
        out = torch.cat(out, dim=-1)
        return out




class ATT(nn.Module):
    def __init__(self, in_dim, t):
        super().__init__()
        self.w = nn.Parameter(torch.rand(in_dim, 1))
        self.b = nn.Parameter(torch.zeros(t, 1))

    def forward(self, x):
        # print(x.shape, self.b.shape)
        a = torch.matmul(x, self.w) + self.b  # [b,t,1]
        a = torch.softmax(a, dim=1)
        out = x + torch.mul(x, a)  # [b,t,f]
        out = torch.mean(out, dim=1, keepdim=False)  # [b,f]
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop, bidirectional=True):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=drop)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


if __name__ == '__main__':
    pass

