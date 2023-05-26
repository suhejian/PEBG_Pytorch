"""
pebg 预训练pytorch复现

Usage:
python pebg.py --dataset=assist09
"""

"""
input: 
    pro_skill_sparse.npz
    skill_skill_sparse.npz
    pro_pro_sparse.npz
    pro_feat.npz

output:
    embedding: (num of question, embedding dim)
"""
import torch.nn as nn
import torch

import os
import numpy as np
import math
from scipy import sparse
from pnn import PNN
import argparse

from utils import set_seed

parser = argparse.ArgumentParser(description='args')
# type是要传入的参数的数据类型  help是该参数的提示信息

parser.add_argument("--dataset", type=str, help='choose dataset', default='assist09')
# parser.add_argument("--seed", type=int, help='seed', default=42)
args = parser.parse_args()

# set_seed(args.seed)

# load the datasets
dataset_dir = f"./data_preprocess/{args.dataset}"
pro_skill_coo = sparse.load_npz(os.path.join(dataset_dir, 'pro_skill_sparse.npz')) 
skill_skill_coo = sparse.load_npz(os.path.join(dataset_dir, 'skill_skill_sparse.npz'))
pro_pro_coo = sparse.load_npz(os.path.join(dataset_dir, 'pro_pro_sparse.npz'))
[pro_num, skill_num] = pro_skill_coo.shape
print('problem number %d, skill number %d' % (pro_num, skill_num))
print('pro-skill edge %d, pro-pro edge %d, skill-skill edge %d' % (pro_skill_coo.nnz, pro_pro_coo.nnz, skill_skill_coo.nnz))

pro_skill_dense = torch.from_numpy(pro_skill_coo.toarray()).cuda() #(N, M)
pro_pro_dense = torch.from_numpy(pro_pro_coo.toarray()).cuda() #(N, N)
skill_skill_dense = torch.from_numpy(skill_skill_coo.toarray()).cuda() #(M,M)

pro_feat = np.load(os.path.join(dataset_dir, 'pro_feat.npz'))['pro_feat']    # [pro_diff_feat, auxiliary_target]
pro_feat = torch.from_numpy(pro_feat).cuda()
print('problem feature shape', pro_feat.shape)


# Hyperparameters
diff_feat_dim = pro_feat.shape[1]-1 # D
embed_dim = 64      # node embedding dim in bipartite
hidden_dim = 128    # hidden dim in PNN
keep_prob = 0.5 
lr = 0.001
bs = 256
epochs = 300

# pebg model define
class Pebg(nn.Module):
    def __init__(self, pro_num, skill_num, embed_dim, diff_feat_num, hidden_dim, keep_prob):
        super().__init__()
        self.pro_embed = nn.Embedding(pro_num, embed_dim)
        self.skill_embed = nn.Embedding(skill_num, embed_dim)
        self.diff_feat_embed = nn.Embedding(diff_feat_num, embed_dim)
        self.pnn = PNN(embed_dim, hidden_dim, keep_prob)
        self.apply(self.init_params)

    def init_params(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight.data, mean=0, std=0.1)

    def forward(self, b, e):
        pro_skill_logits = torch.mm(self.pro_embed.weight[b:e], self.skill_embed.weight.T) # (bs, skill_num)
        skill_skill_logits = torch.mm(self.skill_embed.weight, self.skill_embed.weight.T) # (skill_num, skill_num)
        pro_pro_logits = torch.mm(self.pro_embed.weight[b:e], self.pro_embed.weight.T) # (bs, pro_num)
        
        skill_embed = torch.mm(pro_skill_dense[b:e], self.skill_embed.weight) / torch.mean(pro_skill_dense[b:e], axis=1, keepdims=True)
        diff_feat_embed = torch.mm(pro_feat[b:e, :-1], self.diff_feat_embed.weight)
        pro_final_embed, p = self.pnn([self.pro_embed.weight[b:e], skill_embed, diff_feat_embed])
        return pro_skill_logits, skill_skill_logits, pro_pro_logits, p, pro_final_embed

model = Pebg(pro_num, skill_num, embed_dim, diff_feat_dim, hidden_dim, keep_prob).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_steps = int(math.ceil(pro_num/float(bs)))
loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()


for i in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for step in range(train_steps):
        b, e = step * bs, min((step+1)*bs, pro_num)
        pro_skill_logits, skill_skill_logits, pro_pro_logits, p, pro_final_embed = model(b, e)
        
        # pro-skill 
        pro_skill_loss = loss(pro_skill_logits, pro_skill_dense[b:e])
        # skill-skill
        skill_skill_loss = loss(skill_skill_logits, skill_skill_dense)
        # pro-pro
        pro_pro_loss = loss(pro_pro_logits, pro_pro_dense[b:e])
        # # pnn
        pnn_mse = mse_loss(p, pro_feat[b:e,-1])
        
        joint_loss = pnn_mse + pro_skill_loss + skill_skill_loss + pro_pro_loss

        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()
        train_loss += joint_loss.item()
    train_loss /= train_steps
    print(f'epoch: {i}, loss: {train_loss}')
    
    if i in [50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]:
        model.eval()
        with torch.no_grad():
            pro_skill_logits, skill_skill_logits, pro_pro_logits, p, pro_final_embed = model(0, pro_num)
            pro_repre = model.pro_embed.weight.detach().cpu()
            skill_repre = model.skill_embed.weight.detach().cpu()
            pro_final_repre = pro_final_embed.detach().cpu()
        print(f'pro_final shape: {pro_final_repre.shape}')
        save_dataset_dir = f"./pebg_embeddings/{args.dataset}"
        np.savez(os.path.join(save_dataset_dir, f'embedding_{i}.npz'), 
                pro_repre=pro_repre, skill_repre=[], pro_final_repre=pro_final_repre)