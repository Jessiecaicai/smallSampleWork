import math
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import esm
import dataset
import torch
import torch.nn as nn
import time
import param_parser_esm1b
from apex import amp
from apex.parallel import convert_syncbn_model
from torch.utils.data.dataloader import DataLoader

import numpy as np
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(30)

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls",  "avg"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, tokens, outputs):
        last_hidden = outputs
        attention_mask = 1 - tokens.eq(esm1b_alphabet.padding_idx).type_as(outputs)

        if self.pooler_type in ['cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError

if __name__ == '__main__':
    epochs = 30
    batch_size = 64
    temp = 0.05
    uniref50_train_data = dataset.datasets_pretraining.MLMDataset_Uniref50('train')
    uniref50_valid_data = dataset.datasets_pretraining.MLMDataset_Uniref50('val')

    uniref50_train_loader = DataLoader(uniref50_train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                       collate_fn=uniref50_train_data.collate_fn)
    uniref50_valid_loader = DataLoader(uniref50_valid_data, batch_size=batch_size, shuffle=False, num_workers=4,
                                       collate_fn=uniref50_valid_data.collate_fn)

    # esm1b_model, esm1b_alphabet = esm.pretrained.load_model_and_alphabet("../pretrained_models/esm1b_t33_650M_UR50S.pt")

    args = param_parser_esm1b.params_parser()

    esm1b_alphabet = esm.data.Alphabet.from_architecture(args.arch)

    esm1b_model = esm.model.ProteinBertModel(args, esm1b_alphabet)

    # model_path = "../save/pretraining/best_pretrain_uniref50_val.pt"
    # state_dict = torch.load(model_path)
    # state_esm1b = state_dict['model_state_dict']
    # state_optim = state_dict['optim_state_dict']
    # esm1b_model.load_state_dict({k.replace("module.", ""): v for k, v in state_esm1b.items()})

    esm1b_model = esm1b_model.cuda()

    pooler_type = "avg"
    pooler = Pooler(pooler_type).cuda()
    sim = Similarity(temp=temp).cuda()

    criteria = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(esm1b_model.parameters(), lr=1e-5)

    esm1b_model = convert_syncbn_model(esm1b_model)
    esm1b_model, optimizer = amp.initialize(esm1b_model, optimizer, opt_level='O2')

    esm1b_model.train()
    best_loss = 100000
    best_ppl = 100000
    mlm_loss_weight = 0.5
    cl_loss_weight = 0.5
    for epoch in range(epochs):
        train_tic = time.time()
        train_loss = 0
        train_mlm_loss = 0
        train_cl_loss = 0
        train_ppl = 0
        train_step = 0
        for idx, batch in enumerate(uniref50_train_loader):
            esm1b_mask_tokens = batch['masked_ids']
            lm_labels = batch['targets']
            esm1b_mask_tokens, lm_labels = esm1b_mask_tokens.cuda(), lm_labels.cuda()
            result = esm1b_model(esm1b_mask_tokens)
            logits = result['logits']
            MLM_loss = criteria(logits.contiguous().view(-1, len(esm1b_alphabet.all_toks)),
                                lm_labels.contiguous().view(-1))
            step_train_ppl = np.power(2, MLM_loss.item())

            esm1b_cl_tokens = torch.cat((esm1b_mask_tokens, esm1b_mask_tokens), dim=1)
            esm1b_cl_tokens = esm1b_cl_tokens.cuda()
            esm1b_cl_tokens = esm1b_cl_tokens.view(-1, esm1b_cl_tokens.size(-1))
            result_cl = esm1b_model(esm1b_cl_tokens, repr_layers=[6])
            rep = result_cl['representations'][6]
            pooler_output = pooler(tokens=esm1b_cl_tokens, outputs=rep)
            pooler_output = pooler_output.view((-1, 2, pooler_output.size(-1)))
            z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
            cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().cuda()
            CL_loss = criteria(cos_sim, labels)

            Overall_loss = mlm_loss_weight * MLM_loss + cl_loss_weight * CL_loss

            train_mlm_loss += MLM_loss.item()
            train_ppl += step_train_ppl
            train_cl_loss += CL_loss.item()
            train_loss += Overall_loss.item()
            train_step += 1

            optimizer.zero_grad()
            with amp.scale_loss(Overall_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if train_step > 0 and train_step % 1000 == 0:
                print("Step: {} / {} finish. Training Loss: {:.2f}. Training MLM Loss: {:.2f}. Training CL Loss: {:.2f}."
                    " Training PPL: {:.2f}.".format(train_step, len(uniref50_train_loader), train_loss / train_step,
                                                    train_mlm_loss / train_step,
                                                    train_cl_loss / train_step, train_ppl / train_step))
            if train_step > 0 and train_step % 10000 == 0:
                save_data = {"model_state_dict": esm1b_model.state_dict(),
                             "optim_state_dict": optimizer.state_dict(),
                             "epoch": epoch}
                print("Save model! ")
                torch.save(save_data, "../save/cl/best_pretrain_uniref50_cl_process.pt")
        train_toc = time.time()

        esm1b_model.eval()
        val_tic = time.time()
        val_loss = 0
        val_mlm_loss = 0
        val_cl_loss = 0
        val_ppl = 0
        val_step = 0
        for idx, batch in enumerate(uniref50_valid_loader):
            esm1b_mask_tokens = batch['masked_ids']
            lm_labels = batch['targets']
            esm1b_mask_tokens, lm_labels = esm1b_mask_tokens.cuda(), lm_labels.cuda()
            with torch.no_grad():
                result = esm1b_model(esm1b_mask_tokens)
                MLM_loss = criteria(result['logits'].contiguous().view(-1, len(esm1b_alphabet.all_toks)),
                                    lm_labels.contiguous().view(-1))
                step_val_ppl = np.power(2, MLM_loss.item())

                esm1b_cl_tokens = torch.cat((esm1b_mask_tokens, esm1b_mask_tokens), dim=1)
                esm1b_cl_tokens = esm1b_cl_tokens.cuda()
                esm1b_cl_tokens = esm1b_cl_tokens.view(-1, esm1b_cl_tokens.size(-1))
                result_cl = esm1b_model(esm1b_cl_tokens, repr_layers=[6])
                rep = result_cl['representations'][6]
                pooler_output = pooler(tokens=esm1b_cl_tokens, outputs=rep)
                pooler_output = pooler_output.view((-1, 2, pooler_output.size(-1)))
                z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
                cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
                labels = torch.arange(cos_sim.size(0)).long().cuda()
                CL_loss = criteria(cos_sim, labels)

                Overall_loss = mlm_loss_weight * MLM_loss + cl_loss_weight * CL_loss

            val_mlm_loss += MLM_loss.item()
            val_ppl += step_val_ppl
            val_cl_loss += CL_loss.item()
            val_loss += Overall_loss.item()
            val_step += 1
        print("\nStep: {} / {} finish. Validating Loss: {:.2f}. Validating MLM Loss: {:.2f}. Validating CL Loss: {:.2f}. "
            "Validating PPL: {:.2f}.\n".format(val_step, len(uniref50_valid_loader), (val_loss / val_step),
                                               (val_mlm_loss / val_step),
                                               (val_cl_loss / val_step), (val_ppl / val_step)))
        val_toc = time.time()
        val_loss = val_loss / val_step
        val_ppl = val_ppl / val_step
        # if val_loss < best_loss:
        if val_ppl < best_ppl:
            save_data = {"model_state_dict": esm1b_model.state_dict(),
                         "optim_state_dict": optimizer.state_dict(),
                         "epoch": epoch}
            print("Save model! Best val ppl is: {:.2f}.".format(val_ppl))
            torch.save(save_data, "../save/cl/best_pretrain_uniref50_cl_val.pt")
            best_ppl = val_ppl
            # best_loss = val_loss
        print("\nEpoch: {} / {} finish. Training Loss: {:.2f}. Training Time: {:.2f} s. Validating Loss: {:.2f}. Validating Time: {:.2f} s.\n"
              .format(epoch + 1, epochs, train_loss/train_step, (train_toc - train_tic), val_loss, (val_toc - val_tic)))