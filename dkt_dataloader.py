import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as np


def read_user_sequence(filename, n_questions, max_len=100, min_len=3):
    with open(filename, 'r') as f:
        lines = f.readlines()

    y, skill, problem, real_len = [], [], [], []
    index = 0

    while index < len(lines):
        num = eval(lines[index])

        tmp_pro = list(eval(lines[index + 1])[:max_len])
        tmp_pro = [ele + 1 for ele in tmp_pro]

        tmp_skills = list(eval(lines[index + 2])[:max_len])
        tmp_skills = [ele + 1 for ele in tmp_skills]

        tmp_ans = list(eval(lines[index + 3])[:max_len])
        tmp_ans = [ele for ele in tmp_ans]

        for i in range(0, num, max_len):
            pros = tmp_pro[i:min(num, max_len + i)]
            skills = tmp_skills[i:min(num, max_len + i)]
            ans = tmp_ans[i:min(num, max_len + i)]
            cur_len = len(pros)
            if cur_len < min_len:  # 最后一次划分可能小于min_len
                continue
            y.append(torch.tensor(ans))
            skill.append(torch.tensor(skills))
            problem.append(torch.tensor(pros))
        index += 4
    return problem, skill, y


class CustomDataset(Dataset):

    def __init__(self, seq_list, answer_list):
        self.seq_len = torch.tensor([s.shape[0]
                                     for s in seq_list])  # 获取数据真实的长度
        self.pad_seq = pad_sequence(seq_list,
                                    batch_first=True,
                                    padding_value=0)
        self.pad_ans = pad_sequence(answer_list,
                                    batch_first=True,
                                    padding_value=0)

    def __getitem__(self, index):
        return self.pad_seq[index], self.pad_ans[index], self.seq_len[index]

    def __len__(self):
        return len(self.seq_len)


def get_dataloader(dataset, max_steps=200, batch_size=128, mode='problem'):
    data_folder = os.path.join('/data/hejiansu/16-service/sequence_data', dataset)
    problem_hashmap_path = os.path.join(data_folder, 'problem_id_hashmap.json')
    pro_hashmap = json.load(open(problem_hashmap_path, 'r'))
    skill_hashmap_path = os.path.join(data_folder, 'skill_id_hashmap.json')
    skill_hashmap = json.load(open(skill_hashmap_path, 'r'))

    n_questions = len(pro_hashmap)

    pro_train, skill_train, ans_train = read_user_sequence(
        f'{data_folder}/train.txt',
        n_questions=n_questions,
        max_len=max_steps,
        min_len=3)
    pro_val, skill_val, ans_val = read_user_sequence(f'{data_folder}/dev.txt',
                                                     n_questions=n_questions,
                                                     max_len=max_steps,
                                                     min_len=3)
    pro_test, skill_test, ans_test = read_user_sequence(
        f'{data_folder}/test.txt',
        n_questions=n_questions,
        max_len=max_steps,
        min_len=3)
    assert mode in ['problem', 'skill']
    train_dataset = CustomDataset(pro_train, ans_train)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

    val_dataset = CustomDataset(pro_val, ans_val)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    test_dataset = CustomDataset(pro_test, ans_test)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

    return train_data_loader, val_data_loader, test_data_loader