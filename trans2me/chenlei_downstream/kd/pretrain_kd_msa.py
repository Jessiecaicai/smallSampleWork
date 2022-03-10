import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import dataset
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import param_parser_esm1b
from apex import amp
from apex.parallel import convert_syncbn_model
from torch.utils.data.dataloader import DataLoader

import numpy as np
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(30)

def kl_loss(student_logit, teacher_logit, kl_temperature=10):
    kl_div = nn.KLDivLoss(reduction="batchmean")
    return kl_div(
                F.log_softmax(student_logit / kl_temperature, dim=-1),  # log_p[student_logit]
                F.softmax(teacher_logit / kl_temperature, dim=-1),  # p[teacher_logit]
            )

if __name__ == '__main__':
    epochs = 30
    batch_size = 1
    msa_train_data = dataset.datasets_pretraining.MLMDataset_MSA('train')
    msa_valid_data = dataset.datasets_pretraining.MLMDataset_MSA('val')

    msa_train_loader = DataLoader(msa_train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                  collate_fn=msa_train_data.collate_fn)
    msa_valid_loader = DataLoader(msa_valid_data, batch_size=batch_size, shuffle=False, num_workers=4,
                                  collate_fn=msa_valid_data.collate_fn)

    msa_model, msa_alphabet = esm.pretrained.load_model_and_alphabet("../pretrained_models/esm_msa1_t12_100M_UR50S.pt")

    msa_model = msa_model.cuda()

    args = param_parser_esm1b.params_parser()

    esm1b_alphabet = esm.data.Alphabet.from_architecture(args.arch)

    esm1b_model = esm.model.ProteinBertModel(args, esm1b_alphabet)

    # model_path = "../save/kd/best_kd_msa_val.pt"
    # state_dict = torch.load(model_path)
    # state_esm1b = state_dict['model_state_dict']
    # state_optim = state_dict['optim_state_dict']
    # esm1b_model.load_state_dict({k.replace("module.", ""): v for k, v in state_esm1b.items()})

    esm1b_model.cuda()

    criteria = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(esm1b_model.parameters(), lr=1e-5)

    msa_model = convert_syncbn_model(msa_model)
    msa_model = amp.initialize(msa_model, opt_level='O0')

    esm1b_model = convert_syncbn_model(esm1b_model)
    esm1b_model, optimizer = amp.initialize(esm1b_model, optimizer, opt_level='O0')

    msa_model.eval()
    esm1b_model.train()
    best_loss = 100000
    best_ppl = 100000
    mlm_loss_weight = 0.5
    kd_loss_weight = 0.5
    for epoch in range(epochs):
        train_tic = time.time()
        train_loss = 0
        train_mlm_loss = 0
        train_kd_loss = 0
        train_ppl = 0
        train_step = 0
        for idx, batch in enumerate(msa_train_loader):
            msa_mask_tokens = batch['masked_msa_id']
            lm_labels = batch['masked_msa_labels']
            bs, num_seq, seq_length = msa_mask_tokens.size()
            msa_mask_tokens, lm_labels = msa_mask_tokens.cuda(), lm_labels.cuda()
            esm_mask_tokens, esm_lm_labels = msa_mask_tokens.squeeze(0), lm_labels.squeeze(0)
            # esm_mask_tokens, esm_lm_labels = msa_mask_tokens.view(bs * num_seq, -1), lm_labels.squeeze(0).view(bs * num_seq, -1)
            esm1b_result = esm1b_model(esm_mask_tokens)
            esm1b_logits = esm1b_result['logits']
            MLM_loss = criteria(esm1b_logits.contiguous().view(-1, len(esm1b_alphabet.all_toks)),
                                esm_lm_labels.contiguous().view(-1))
            step_train_ppl = np.power(2, MLM_loss.item())
            with torch.no_grad():
                msa_result = msa_model(msa_mask_tokens)
                msa_logits = msa_result['logits']

            KD_loss = kl_loss(esm1b_logits, msa_logits.squeeze(0))
            # KD_loss = kl_loss(esm1b_logits, msa_logits.view(bs * num_seq, seq_length, -1))

            Overall_loss = mlm_loss_weight * MLM_loss + kd_loss_weight * KD_loss

            train_mlm_loss += MLM_loss.item()
            train_ppl += step_train_ppl
            train_kd_loss += KD_loss.item()
            train_loss += Overall_loss.item()

            train_step += 1

            optimizer.zero_grad()
            with amp.scale_loss(Overall_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if train_step > 0 and train_step % 1000 == 0:
                print("Step: {} / {} finish. Training Loss: {:.2f}. Training MLM Loss: {:.2f}. Training KD Loss: {:.2f}."
                      " Training PPL: {:.2f}.".format(train_step, len(msa_train_loader), train_loss/train_step, train_mlm_loss/train_step,
                              train_kd_loss/train_step, train_ppl/train_step))
            if train_step > 0 and train_step % 10000 == 0:
                save_data = {"model_state_dict": esm1b_model.state_dict(),
                             "optim_state_dict": optimizer.state_dict(),
                             "epoch": epoch}
                print("Save model! ")
                torch.save(save_data, "../save/kd/best_kd_msa_process.pt")
        train_toc = time.time()

        esm1b_model.eval()
        val_tic = time.time()
        val_loss = 0
        val_mlm_loss = 0
        val_kd_loss = 0
        val_ppl = 0
        val_step = 0
        for idx, batch in enumerate(msa_valid_loader):
            msa_mask_tokens = batch['masked_msa_id']
            lm_labels = batch['masked_msa_labels']
            bs, num_seq, seq_length = msa_mask_tokens.size()
            msa_mask_tokens, lm_labels = msa_mask_tokens.cuda(), lm_labels.cuda()
            esm_mask_tokens, esm_lm_labels = msa_mask_tokens.squeeze(0), lm_labels.squeeze(0)
            # esm_mask_tokens, esm_lm_labels = msa_mask_tokens.view(bs * num_seq, -1), lm_labels.squeeze(0).view(bs * num_seq, -1)
            with torch.no_grad():
                msa_result = msa_model(msa_mask_tokens)
                msa_logits = msa_result['logits']
                esm1b_result = esm1b_model(esm_mask_tokens)
                esm1b_logits = esm1b_result['logits']
                MLM_loss = criteria(esm1b_logits.contiguous().view(-1, len(esm1b_alphabet.all_toks)),
                                    esm_lm_labels.contiguous().view(-1))
                step_val_ppl = np.power(2, MLM_loss.item())

                KD_loss = kl_loss(esm1b_logits, msa_logits.squeeze(0))
                # KD_loss = kl_loss(esm1b_logits, msa_logits.view(bs * num_seq, seq_length, -1))

                Overall_loss = mlm_loss_weight * MLM_loss + kd_loss_weight * KD_loss

            val_mlm_loss += MLM_loss.item()
            val_ppl += step_val_ppl
            val_kd_loss += KD_loss.item()
            val_loss += Overall_loss.item()

            val_step += 1

        print("\nStep: {} / {} finish. Validating Loss: {:.2f}. Validating MLM Loss: {:.2f}. Validating KD Loss: {:.2f}. "
            "Validating PPL: {:.2f}.\n".format(val_step, len(msa_valid_loader), (val_loss / val_step), (val_mlm_loss / val_step),
                   (val_kd_loss / val_step), (val_ppl / val_step)))
        val_toc = time.time()
        val_loss = val_loss / val_step
        val_ppl = val_ppl / val_step
        # if val_loss < best_loss:
        if val_ppl < best_ppl:
            save_data = {"model_state_dict": esm1b_model.state_dict(),
                         "optim_state_dict": optimizer.state_dict(),
                         "epoch": epoch}
            print("Save model! Best val ppl is: {:.2f}.".format(val_ppl))
            torch.save(save_data, "../save/kd/best_kd_msa_val.pt")
            best_ppl = val_ppl
            # best_loss = val_loss
        print("\nEpoch: {} / {} finish. Training Loss: {:.2f}. Training Time: {:.2f} s. Validating Loss: {:.2f}. Validating Time: {:.2f} s.\n"
              .format(epoch + 1, epochs, train_loss/train_step, (train_toc - train_tic), val_loss, (val_toc - val_tic)))