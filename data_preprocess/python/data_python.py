import os 
import pandas as pd
import numpy as np
from scipy import sparse
import json
from tqdm import tqdm

class DataProcess():
    def __init__(self, data_folder='python', file_name='python.csv', min_inter_num=3):
        print("Process Dataset %s" % data_folder)
        self.min_inter_num = min_inter_num
        self.data_folder = data_folder
        self.file_name = file_name

    def pro_skill_graph(self):
        pro_id_dict_path = '/data/hejiansu/16-service/sequence_data/python/problem_id_hashmap.json'
        skill_id_dict_path = '/data/hejiansu/16-service/sequence_data/python/skill_id_hashmap.json'
        file = '/data/hejiansu/16-service/sequence_data/python/python.csv'

        df = pd.read_csv(file, low_memory=False, encoding="ISO-8859-1")
        
        with open(pro_id_dict_path, 'r') as f:
            pro_id_dict = json.load(f)
        with open(skill_id_dict_path, 'r') as f:
            skill_id_dict = json.load(f)
        
        print('problem number %d' % len(pro_id_dict))
        print('skill number %d' % len(skill_id_dict))

        pro_diff = df['difficulty'].unique()
        pro_diff_dict = dict(zip(pro_diff, range(len(pro_diff))))
        print('problem diff level: ', pro_diff_dict)

        pro_feat = [] 
        pro_skill_adj = []
        
        for pro, pro_id in tqdm(pro_id_dict.items()):
            
            tmp_df = df[df['problem_id']==int(pro)]
            assert len(tmp_df) > 0
            tmp_df_0 = tmp_df.iloc[0]

            # pro_feature: [ref_count, pro_diff, mean_correct_num]
            ref_count = tmp_df_0.reference_count
            p = (tmp_df.status == 'ACCEPTED').mean() 
            pro_diff_id = pro_diff_dict[tmp_df_0['difficulty']] 
            tmp_pro_feat = [0.] * (len(pro_diff_dict)+2)
            tmp_pro_feat[0] = ref_count
            tmp_pro_feat[pro_diff_id+1] = 1.
            tmp_pro_feat[-1] = p
            pro_feat.append(tmp_pro_feat)

            # build problem-skill bipartite
            tmp_skills = [ele for ele in tmp_df_0['skill_id'].split('_')]
            assert len(tmp_skills) > 0
            for s in tmp_skills:
                pro_skill_adj.append([pro_id, skill_id_dict[s], 1])

        pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
        pro_feat = np.array(pro_feat).astype(np.float32)
        pro_feat[:, 0] = (pro_feat[:, 0] - np.min(pro_feat[:, 0])) / (np.max(pro_feat[:, 0])-np.min(pro_feat[:, 0]))
        pro_num = np.max(pro_skill_adj[:, 0]) + 1
        skill_num = np.max(pro_skill_adj[:, 1]) + 1
        # import ipdb
        # ipdb.set_trace()
        print('problem number %d, skill number %d' % (pro_num, skill_num))

        # save pro-skill-graph in sparse matrix form
        pro_skill_sparse = sparse.coo_matrix((pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])), shape=(pro_num, skill_num))
        sparse.save_npz(os.path.join(self.data_folder, 'pro_skill_sparse.npz'), pro_skill_sparse)

        # save pro-id-dict, skill-id-dict
        self.save_dict(pro_id_dict, os.path.join(self.data_folder, 'pro_id_dict.txt'))
        self.save_dict(skill_id_dict, os.path.join(self.data_folder, 'skill_id_dict.txt'))

        # save pro_feat_arr
        np.savez(os.path.join(self.data_folder, 'pro_feat.npz'), pro_feat=pro_feat)


    def save_dict(self, dict_name, file_name):
        f = open(file_name, 'w')
        f.write(str(dict_name))
        f.close

    def write_txt(self, file, data):
        with open(file, 'w') as f:
            for dd in data:
                for d in dd:
                    f.write(str(d)+'\n')

if __name__ == '__main__':
    data_folder = '.'
    file_name='/data/hejiansu/16-service/sequence_data/python/python.csv'

    DP = DataProcess(data_folder, file_name)

    # excute the following function step by step
    DP.pro_skill_graph() # finish
