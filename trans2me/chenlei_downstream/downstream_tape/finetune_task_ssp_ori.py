import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import esm
import model
import torch
import os
import dataset
import time
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import numpy as np
import scipy.stats
from apex import amp
from apex.parallel import convert_syncbn_model

import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(30)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def mean_squared_error(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))


def mean_absolute_error(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.mstats.spearmanr(target_array, prediction_array).correlation


def accuracy(target, prediction):
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total


if __name__ == '__main__':
    data_path = '../data/downstream'
    epochs = 30
    batch_size = 8
    ss_train_data = dataset.datasets_downstream_tape.SecondaryStructureDataset(data_path, 'train')
    ss_valid_data = dataset.datasets_downstream_tape.SecondaryStructureDataset(data_path, 'valid')
    ss_test1_data = dataset.datasets_downstream_tape.SecondaryStructureDataset(data_path, 'casp12')
    ss_test2_data = dataset.datasets_downstream_tape.SecondaryStructureDataset(data_path, 'ts115')
    ss_test3_data = dataset.datasets_downstream_tape.SecondaryStructureDataset(data_path, 'cb513')

    ss_train_loader = DataLoader(
        ss_train_data, batch_size=batch_size, shuffle=True, collate_fn=ss_train_data.collate_fn
    )
    ss_valid_loader = DataLoader(
        ss_valid_data, batch_size=batch_size, shuffle=False, collate_fn=ss_valid_data.collate_fn
    )
    ss_test3_loader = DataLoader(
        ss_valid_data, batch_size=batch_size, shuffle=False, collate_fn=ss_test3_data.collate_fn
    )

    downstream_model = model.model_down_tape.ProteinBertForSequenceToSequenceClassification().cuda()

    optimizer = torch.optim.AdamW(downstream_model.parameters(), lr=1e-5)

    downstream_model = convert_syncbn_model(downstream_model)
    downstream_model, optimizer = amp.initialize(downstream_model, optimizer, opt_level='O0')

    best_loss = 100000
    best_acc = 0
    downstream_model.train()
    for epoch in range(epochs):
        train_tic = time.time()
        train_loss = 0
        train_acc = 0
        train_step = 0
        for idx, batch in enumerate(ss_train_loader):
            ss_inputs = batch['input_ids']
            ss_targets = batch['targets']
            ss_inputs, ss_targets = ss_inputs.cuda(), ss_targets.cuda()
            outputs = downstream_model(ss_inputs, targets=ss_targets)
            loss_acc, value_prediction = outputs
            loss = loss_acc[0]
            acc = loss_acc[1]['accuracy']
            # acc = accuracy(ss_targets.detach().cpu().numpy(), value_prediction.detach().cpu().numpy())

            train_loss += loss.item()
            train_acc += acc.item()
            train_step += 1

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if train_step > 0 and train_step % 100 == 0:
                print("Step: {} / {} finish. Training Loss: {:.2f}. Training Accuracy: {:.2f}."
                      .format(train_step, len(ss_train_loader), (train_loss / train_step), (train_acc / train_step)))

        train_toc = time.time()

        downstream_model.eval()
        val_tic = time.time()
        val_loss = 0
        val_acc = 0
        val_step = 0
        for idx, batch in enumerate(ss_valid_loader):
            ss_inputs = batch['input_ids']
            ss_targets = batch['targets']
            ss_inputs, ss_targets = ss_inputs.cuda(), ss_targets.cuda()
            with torch.no_grad():
                outputs = downstream_model(ss_inputs, targets=ss_targets)
                loss_acc, value_prediction = outputs
                loss = loss_acc[0]
                acc = loss_acc[1]['accuracy']
                # acc = accuracy(ss_targets.detach().cpu().numpy(), value_prediction.detach().cpu().numpy())

            val_loss += loss.item()
            val_acc += acc.item()
            val_step += 1

        print("\nStep: {} / {} finish. Validating Loss: {:.2f}. Validating Accuracy: {:.2f}.\n".
              format(val_step, len(ss_valid_loader), (val_loss / val_step), (val_acc / val_step)))
        val_toc = time.time()
        val_loss = val_loss / val_step
        val_acc = val_acc / val_step
        # if val_loss < best_loss:
        if val_acc > best_acc:
            save_data = {"model_state_dict": downstream_model.state_dict(),
                         "optim_state_dict": optimizer.state_dict(),
                         "epoch": epoch}
            print("Save model! Best val Accuracy is: {:.2f}.".format(val_acc))
            torch.save(save_data, "../save/downstream/best_ssp_ori.pt")
            best_acc = val_acc
            # best_loss = val_loss
        print("\nEpoch: {} / {} finish. Training Loss: {:.2f}. Training Time: {:.2f} s. Validating Loss: {:.2f}. Validating Time: {:.2f} s.\n"
            .format(epoch + 1, epochs, train_loss / train_step, (train_toc - train_tic), val_loss, (val_toc - val_tic)))

    # data_path = '../data/downstream'
    # batch_size = 8
    # ss_test_data = dataset.datasets_downstream_tape.SecondaryStructureDataset(data_path, 'cb513')
    # ss_test_loader = DataLoader(
    #     ss_test_data, batch_size=batch_size, shuffle=False, collate_fn=ss_test_data.collate_fn,
    # )
    # model_path = "../save/downstream/best_ssp_ori.pt"
    # state_dict = torch.load(model_path)
    # state_ss = state_dict['model_state_dict']
    # model = model.model_down_tape.ProteinBertForSequenceToSequenceClassification()
    # # model.load_state_dict(state_ss)
    # model.load_state_dict({k.replace("module.", ""): v for k, v in state_ss.items()})
    # for k, v in model.named_parameters():
    #     v.requires_grad = False
    # model.eval().cuda()
    # test_loss = 0
    # test_acc = 0
    # test_step = 0
    # test_tic = time.time()
    # for idx, batch in enumerate(ss_test_loader):
    #     ss_inputs = batch['input_ids']
    #     ss_targets = batch['targets']
    #     ss_inputs, ss_targets = ss_inputs.cuda(), ss_targets.cuda()
    #     with torch.no_grad():
    #         outputs = model(ss_inputs, targets=ss_targets)
    #         loss_acc, value_prediction = outputs
    #         loss = loss_acc[0]
    #         acc = loss_acc[1]['accuracy']
    #
    #     test_loss += loss.item()
    #     test_acc += acc.item()
    #     test_step += 1
    #
    # test_toc = time.time()
    #
    # print("Step: {} / {} finish. Test Loss: {:.2f}. Test Accuracy: {:.2f}. Test Time: {:.2f}.".
    #       format(test_step, len(ss_test_loader), (test_loss / test_step), (test_acc / test_step),
    #              (test_toc - test_tic)))