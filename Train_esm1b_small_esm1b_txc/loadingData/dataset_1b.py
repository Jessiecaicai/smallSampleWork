# -*- coding: utf-8 -*-
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from random import *
from scipy.spatial.distance import pdist, squareform

from .tokenizers import Tokenizer

logger = logging.getLogger(__name__)

class Configue(object):
    label_list = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
    max_mask = 100 # max number of padding

def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")

def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item

# task('fluorescence')
class FluorescenceDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, Tokenizer] = 'iupac',
                 in_memory: bool = False):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'fluorescence/fluorescence_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        seq_length = item['protein_length']
        # if seq_length <= 256:
        #     item['primary'] = item['primary']
        # else:
        #     item['primary'] = item['primary'][:256]
        # token_ids = self.tokenizer.encode(item['primary'])
        # input_mask = np.ones_like(token_ids)
        # return token_ids, input_mask, float(item['log_fluorescence'][0])
        return item['primary']
    # def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
    #     input_ids, input_mask, fluorescence_true_value = tuple(zip(*batch))
    #     input_ids = torch.from_numpy(pad_sequences(input_ids, 1))
    #     input_mask = torch.from_numpy(pad_sequences(input_mask, 1))
    #     fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore
    #     fluorescence_true_value = fluorescence_true_value.unsqueeze(1)
    #
    #     return {'input_ids': input_ids,
    #             'input_mask': input_mask,
    #             'targets': fluorescence_true_value}
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
            mask_number = min(Configue.max_mask,max(1,int(len(batch[i])*0.15)))
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
            tokens[i,1:len(batch[i]) + 1] = seq
            tokens[i, len(batch[i]) + 1] = Configue.label_list.index("<eos>")

            all_mask_tokens.append(masked_tokens)

        all_label_ids = torch.LongTensor(all_mask_tokens)

        return tokens, all_label_ids