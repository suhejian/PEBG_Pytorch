"""
Pnn

question embed:     q
average skill:      s
attribute feature:  a

input:  Z = (z1, z2, z3) = (q, s, a)
        P = [pij], pij=<zi, zj>

transform 2 info matrix -> signal vector lz, lp

l_z^k = sum(W_z^k * Z)
l_p^k = sum(W_P ^k * P)

W_p^k = theta @ theta.T

e = ReLU(l_z + l_p + b)

"""

import torch
import torch.nn as nn

class PNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, keep_prob):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim*3 + 3, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=keep_prob)
        self.embed_dim = embed_dim
        
        
    def forward(self, inputs):
        num_inputs = len(inputs)
        num_pairs = int(num_inputs * (num_inputs-1) / 2)
        xw = torch.cat(inputs, 1) #(bs, ,embed_sim)
        xw3d = xw.reshape(-1, num_inputs, self.embed_dim) #(bs, 3, embed_dim)

        row, col = [], []
        for i in range(num_inputs-1):
            for j in range(i+1, num_inputs):
                row.append(i)
                col.append(j)
       
        p = xw3d.permute(1, 0, 2)[row].permute(1, 0, 2)
                 # (bs, pair, embed_dim)
        q = xw3d.permute(1, 0, 2)[col].permute(1, 0, 2) # (bs, pair, embed_dim)
        ip = torch.sum(p * q, -1).reshape(-1, num_pairs)
        # import ipdb; ipdb.set_trace()
        l = torch.cat([xw, ip], 1)
        # import ipdb; ipdb.set_trace()
        h = self.act(self.linear1(l))
        h = self.dropout(h)
        p = self.linear2(h).reshape(-1)
        return h, p

from utils import set_seed

# if __name__ == '__main__':
#     set_seed(42)
#     bs = 4
#     embed_dim = 5
#     q = torch.randn(bs, embed_dim)
#     s = torch.randn(bs, embed_dim)
#     a = torch.randn(bs, embed_dim)
#     model = PNN(embed_dim, 6, 0.5)
#     model.train()
#     out = model([q,s,a])
#     import ipdb; ipdb.set_trace()