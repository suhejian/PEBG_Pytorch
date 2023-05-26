import os 
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from scipy import sparse


class DataProcess():
    def __init__(self, data_folder='assist09', file_name='assist09.csv', min_inter_num=3):
        print("Process Dataset %s" % data_folder)
        self.min_inter_num = min_inter_num
        self.data_folder = data_folder
        self.file_name = file_name

    def pro_skill_graph(self):
        with open(os.path.join(self.data_folder, 'problem_id_hashmap.json'), 'r') as f:
            pro_id_dict = json.load(f)
        with open(os.path.join(self.data_folder, 'skill_id_hashmap.json'), 'r') as f:
            skill_id_dict = json.load(f)
        df = pd.read_csv(os.path.join(self.data_folder, file_name),low_memory=False, encoding="ISO-8859-1")
   
        print('problem number %d' % len(pro_id_dict))
        print('skill number %d' % len(skill_id_dict))

        pro_type = df['answer_type'].unique()
        pro_type_dict = dict(zip(pro_type, range(len(pro_type))))
        print('problem type: ', pro_type_dict)

        pro_feat = []
        pro_skill_adj = []
        
        for pro_id in tqdm(pro_id_dict.keys()):          
            tmp_df = df[df['problem_id']==int(pro_id)]
            assert len(tmp_df) > 0
            tmp_df_0 = tmp_df.iloc[0]

            # pro_feature: [ms_of_response, answer_type, mean_correct_num]
            ms = tmp_df['ms_first_response'].abs().mean()
            p = tmp_df['correct'].mean()
            pro_type_id = pro_type_dict[tmp_df_0['answer_type']] 
            tmp_pro_feat = [0.] * (len(pro_type_dict)+2)
            tmp_pro_feat[0] = ms
            tmp_pro_feat[pro_type_id+1] = 1.
            tmp_pro_feat[-1] = p
            pro_feat.append(tmp_pro_feat)

            # build problem-skill bipartite
            tmp_skills = [ele for ele in tmp_df_0['skill_id'].split('_')]
            for s in tmp_skills:
                assert s in skill_id_dict, s
                pro_skill_adj.append([pro_id_dict[pro_id], skill_id_dict[s], 1])
        # import ipdb; ipdb.set_trace()
        pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
        pro_feat = np.array(pro_feat).astype(np.float32)
        pro_feat[:, 0] = (pro_feat[:, 0] - np.min(pro_feat[:, 0])) / (np.max(pro_feat[:, 0])-np.min(pro_feat[:, 0]))
        pro_num = np.max(pro_skill_adj[:, 0]) + 1
        skill_num = np.max(pro_skill_adj[:, 1]) + 1
        print('problem number %d, skill number %d' % (pro_num, skill_num))

        # save pro-skill-graph in sparse matrix form
        pro_skill_sparse = sparse.coo_matrix((pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])), shape=(pro_num, skill_num))
        sparse.save_npz('pro_skill_sparse.npz', pro_skill_sparse)

        # save pro-id-dict, skill-id-dict
        self.save_dict(pro_id_dict, 'pro_id_dict.txt')
        self.save_dict(skill_id_dict, 'skill_id_dict.txt')

        # save pro_feat_arr
        np.savez('pro_feat.npz', pro_feat=pro_feat)

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
    # 存放assist09数据集的目录, 里面包括干净csv文件和hashmap
    data_folder = '/data/hejiansu/16-service/sequence_data/assist09' 
    min_inter_num = 3
    file_name= 'assist09.csv'

    DP = DataProcess(data_folder, file_name, min_inter_num)
    DP.pro_skill_graph()
    