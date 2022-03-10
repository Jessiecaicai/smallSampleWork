# -*-coding:utf-8-*-
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
from random import *

class Configue(object):
    label_list = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
    max_mask = 100 # max number of padding

class getUniref50(Dataset):
    def __init__(self):
        self.uniref50_txt_location = "/home/guo/data/datacluster/uniref50/db/uniref50_256.txt"
        # self.uniref50_txt_location = "/home/guo/data/datacluster/11.txt"
        self.data = self.getSequenceData()
        self.len = len(self.data)

    # def get_file_location(self):  # get filename
    #     dir_file = []
    #     s = "/home/guo/data/datacluster/uniref50/db/uniref50.fasta"
    #     # 改变当前路径到/home/guo/data/datacluster/uniref50/db
    #     os.chdir(s)
    #     files = glob.glob('*.txt')
    #     for iter, filename in enumerate(files):
    #         dir_file.append(filename)
    #     return dir_file

    def getSequenceData(self):
        filename_location = self.uniref50_txt_location
        data = np.loadtxt(filename_location, dtype=list).tolist()  # type from array to list
        return data
    # 可以将高负载的操作放在这里，因为多进程时会并行调用这个函数，这样可以加速
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    # def process_data_mask(self):
    # 采用动态mask的方式
    @staticmethod
    def collate__fn(batch):

        all_mask_tokens = []
        max_len_data = max(len(batch[i]) for i in range(len(batch)))
        batch_size = len(batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len_data + 2  # 填充<cls>和<eos>
            ),
            dtype=torch.int64,
        )
        tokens.fill_(Configue.label_list.index('<pad>'))

        for i in range(batch_size):
            tokens[i,0] = Configue.label_list.index("<cls>")

            label = []
            # 在训练的时候忽略-1的数据
            masked_tokens = np.zeros([max_len_data + 2], np.int64) - 1

            for item in batch[i]:
                label.append(Configue.label_list.index(item))  # 真实label
            # 设置mask的数量
            mask_number = min(Configue.max_mask, max(1, int(len(batch[i])*0.15)))
            cand_mask_pos = [i for i, token in enumerate(batch[i])]  # [0,1,2,3,4..]
            shuffle(cand_mask_pos)

            for pos in cand_mask_pos[:mask_number]:
                masked_tokens[pos] = label[pos]     # 真实的label
                if random() < 0.8:                  # 80% 填充mask
                    label[pos] = Configue.label_list.index('<mask>')
                elif random() > 0.9:                # 10% 填充为其他
                    index = randint(4, 28)
                    label[pos] = index

            seq = torch.tensor(label)
            tokens[i, 1:len(batch[i]) + 1] = seq
            tokens[i, len(batch[i]) + 1] = Configue.label_list.index("<eos>")

            all_mask_tokens.append(masked_tokens)

        all_label_ids = torch.LongTensor(all_mask_tokens)

        return tokens, all_label_ids