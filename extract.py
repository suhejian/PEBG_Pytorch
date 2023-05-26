import os
import sys
import numpy as np
from scipy import sparse
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',type=str)

args = parser.parse_args()
# 之前得到的`pro_feat.npz`和`pro_skill_sparse.npz`文件位置
data_folder = f"./data_preprocess/{args.dataset}"

pro_skill_coo = sparse.load_npz(os.path.join(data_folder, 'pro_skill_sparse.npz'))
[pro_num, skill_num] = pro_skill_coo.toarray().shape
print('problem number %d, skill number %d' % (pro_num, skill_num))
pro_skill_csc = pro_skill_coo.tocsc()
pro_skill_csr = pro_skill_coo.tocsr()


def extract_pro_pro_sim():
    # extract pro-pro similarity sparse matrix
    pro_pro_adj = []
    for p in tqdm(range(pro_num), desc='extract pro-pro similarity sparse matrix'):
        tmp_skills = pro_skill_csr.getrow(p).indices
        similar_pros = pro_skill_csc[:, tmp_skills].indices
        zipped = zip([p] * similar_pros.shape[0], similar_pros)
        pro_pro_adj += list(zipped)

    pro_pro_adj = list(set(pro_pro_adj))
    pro_pro_adj = np.array(pro_pro_adj).astype(np.int32)
    data = np.ones(pro_pro_adj.shape[0]).astype(np.float32)
    pro_pro_sparse = sparse.coo_matrix((data, (pro_pro_adj[:, 0], pro_pro_adj[:, 1])), shape=(pro_num, pro_num))
    sparse.save_npz(os.path.join(data_folder, 'pro_pro_sparse.npz'), pro_pro_sparse)


def extract_skill_skill_sim():
    # extract skill-skill similarity sparse matrix
    skill_skill_adj = []
    for s in tqdm(range(skill_num), desc='extract skill-skill similarity sparse matrix'):
        tmp_pros = pro_skill_csc.getcol(s).indices
        similar_skills = pro_skill_csr[tmp_pros, :].indices
        zipped = zip([s] * similar_skills.shape[0], similar_skills)
        skill_skill_adj += list(zipped)

    skill_skill_adj = list(set(skill_skill_adj))
    skill_skill_adj = np.array(skill_skill_adj).astype(np.int32)
    data = np.ones(skill_skill_adj.shape[0]).astype(np.float32)
    skill_skill_sparse = sparse.coo_matrix((data, (skill_skill_adj[:, 0], skill_skill_adj[:, 1])), shape=(skill_num, skill_num))
    sparse.save_npz(os.path.join(data_folder, 'skill_skill_sparse.npz'), skill_skill_sparse)


if __name__=="__main__":

    extract_pro_pro_sim()
    extract_skill_skill_sim()