from typing import Union, List, Tuple, Sequence, Dict, Any
from copy import copy
from collections import OrderedDict
import random
import numpy as np
from einops import repeat
import torch
from torch.utils.data import Dataset
from .tokenizers import Tokenizer

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

class MLMDataset_Uniref50(Dataset):

    def __init__(self,
                 split: str,
                 tokenizer: Union[str, Tokenizer] = 'iupac'):
        super().__init__()
        if split not in ('train', 'val'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'val']")
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        # data_file = f'../data/pretrain/uniref50_shuffle_{split}.txt'
        data_file = f'/research/zqg/dataset/uniref50_{split}.txt'
        data = open(data_file, 'r', encoding='utf-8')
        self.data = data.readlines()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        item = item.replace('\n', '')
        seq_length = len(item)
        # tokens = self.tokenizer.tokenize(item)
        # 取256长度进行预训练 tokens [ETDGVFGDDG]
        if seq_length <= 256:
            tokens = self.tokenizer.tokenize(item)
        else:
            tokens = self.tokenizer.tokenize(item[:256])

        # tokens [<cls>ETDGVFGDDG<eos>]
        tokens = self.tokenizer.add_special_tokens(tokens)

        # ori_tokens [<cls>ETDGVFGDDG<eos>]
        ori_tokens = copy(tokens)

        # ori_token_ids [074683652]
        ori_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(ori_tokens), np.int64)

        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        # ori_tokens_ids 为最原始tokens_ids
        # masked_tokens_ids 为mask之后的tokens_ids
        # labels 标记mask的地方 mask的数字为ori_ids 不动的为-1
        return ori_token_ids, masked_token_ids, labels

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        ori_ids, masked_ids, lm_label_ids = tuple(zip(*batch))

        ori_ids = torch.from_numpy(pad_sequences(ori_ids, 1))
        masked_ids = torch.from_numpy(pad_sequences(masked_ids, 1))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

        return {'ori_ids': ori_ids,
                'masked_ids': masked_ids,
                'targets': lm_label_ids}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                pass

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels