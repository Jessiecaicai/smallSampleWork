# -*-coding:utf-8-*-
import os
import argparse
import numpy as np
import torch
from apex.parallel.multiproc import world_size
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.data.distributed import DistributedSampler
from apex.parallel import convert_syncbn_model
import param_esm1b
import loadingData
import esmz

# DDP 多机多卡需要保证初始化的模型相同
def init_seeds(SEED=1):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False # AMP requires torch.backends.cudnn.enabled = True
    # 实验表明不设置这个模型初始化并不会受到影响
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

args = param_esm1b.params_parser()

# 3）设置cuda
# local_rank = torch.distributed.get_rank()
#torch.cuda.set_device(args.local_rank)
# device = torch.device("cuda",args.local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 初始化参数
    # args_model = param_esm1b.params_parser()
    esm1b_alphabet = esmz.data.Alphabet.from_architecture(args.arch)
    model = esmz.model.ProteinBertModel(args, esm1b_alphabet)

    exit()






































