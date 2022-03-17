from pandas.tests.tools.test_to_datetime import epochs
from scipy.io.arff.tests.test_arffread import data_path
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import esm
import esmz
import torch
# torch.set_printoptions(profile="full")
import param_esm1b
import torch.distributed
import numpy as np
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
from apex import amp
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import DataParallel
import loadingData
import torch.nn as nn
import pytest

### 计算模型参数量
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

### 得到mask的ACC
def compute_accuracy(pred, target):
    #return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).item()) / len(pred)

### 训练模型
### 保证初始化的模型相同
def init_seeds(SEED=1):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

### 对所有GPU上的loss求平均，打印输出
def reduce_loss(value, average=True):
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        output_tensors = value.clone()
        torch.distributed.all_reduce(output_tensors)
        if average:
            output_tensors /= world_size
        return output_tensors

# 1) 多机多卡的初始化
torch.distributed.init_process_group(backend='nccl')

# 2) 从外获得local_rank的参数，多机多卡的torch.distributed.launch会传给这个参数
# args = param_esm1b.params_parser()
local_rank = torch.distributed.get_rank()

# 3） 设置cuda
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if __name__ == '__main__':

    ### 初始化参数
    epochs = 20000
    batch_size = 300
    learning_rate =5e-5
    Seed = 2022
    init_seeds(SEED=Seed)

    ### 加载训练参数
    # data_path = '/home/guo/data/datacluster/uniref50/db/uniref50.fasta'
    # train_dataset = loadingData.getUniref50(data_path).FluorescenceDataset(data_path, 'train')
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=train_dataset.collate__fn)
    train_dataset = loadingData.write_dataset_1b_uniref50.MLMDataset_Uniref50('train')
    train_sample = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                              drop_last=True,
                              collate_fn=train_dataset.collate_fn,
                              sampler=train_sample,
                              num_workers=3)

    ### 加载原esm1b模型参数 33层
    ### 随机初始化参数
    args = param_esm1b.params_parser()
    esm1b_alphabetAfter = esm.data.Alphabet.from_architecture(args.arch)
    # model = esmz.model.ProteinBertModel(args, esm1b_alphabetAfter)

    model = esm.model.ProteinBertModel(args, esm1b_alphabetAfter)
    # model, alphabet = esm.pretrained.load_model_and_alphabet("/research/wzy/esm1b/esm1b_t33_650M_UR50S.pt")
    # esm1b, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    # state_dict = torch.load(model)
    # state_esm1b = state_dict['model_state_dict']
    ### 赋原1b模型参数
    # model.state_dict().update({k: v for k, v in esm1b.state_dict().items() if k in model.state_dict().keys()})  # k为参数名 v对应参数值

    # device_ids = [0, 1]
    # model = DataParallel(model)
    Total, Trainable = get_parameter_number(model)
    print(f"Total:{Total}, Trainable:{Trainable}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # 在使用nn.DistributedDataParallel时，用nn.SyncBatchNorm替换或包装nn.BatchNorm层。
    model = convert_syncbn_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    model.to(device)

    model.train()
    for epoch_item in range(epochs):
        train_loader.sampler.set_epoch(epoch_item)
        training_loss = 0
        training_acc = 0
        training_step = 0
        training_step_out = 0
        for i, data in enumerate(train_loader):
            tokens = data["masked_ids"]
            all_label_ids = data["targets"]
            tokens, all_label_ids = tokens.cuda(), all_label_ids.cuda()

            results = model(tokens, return_contacts=False)
            logits = results["logits"].cuda()

            loss = criterion(logits.contiguous().view(-1, len(esm1b_alphabetAfter.all_toks)),
                             all_label_ids.contiguous().view(-1))

            mask_idex = []
            mask_item = []
            for j, index in enumerate(all_label_ids.contiguous().view(-1)):
                if index != -1:
                    mask_idex.append(j)
                    mask_item.append(index)
            pre_atom = logits.contiguous().view(-1, len(esm1b_alphabetAfter.all_toks))[mask_idex]
            acc = compute_accuracy(pre_atom.cpu(), torch.tensor(mask_item).cpu())
            # print(f"acc:{acc}")
            training_acc += acc

            training_loss += loss
            training_step += 1
            training_step_out += 1

            optimizer.zero_grad()
            with amp.scale_loss(loss,optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            optimizer.step()

            if training_step_out % 100 == 0:
                training_loss /= training_step_out
                reduceLOSS = reduce_loss(training_loss)
                # 在每台机器上都输出loss
                if local_rank == 0:
                    training_acc /= training_step_out
                    print(
                        "Epoch: {}. \t Step: {} / {} finish. \t TrainingLoss: {} \t TrainingAcc: {}".format(epoch_item,
                                                                                                            training_step,
                                                                                                            len(train_loader),
                                                                                                            training_loss,
                                                                                                            training_acc))
                    loss_txt = open("./loss/loss2022_0317_loss.txt", 'a')
                    loss_txt.write("Epoch: {}. \t Step: {} / {} finish. \t TrainingLoss: {}  \t TrainingAcc: {} \n".format(epoch_item, training_step, len(train_loader), training_loss, training_acc))
                training_loss = 0
                training_step_out = 0
                training_acc = 0


            if i % 2000 == 0:                         # 迭代10000次保存一次模型
                if torch.distributed.get_rank() == 0:   # 在第一台机器的第一张卡上保存模型
                    model_path = os.path.join("./", "317model_" + str(epoch_item) + "_" + str(i) + ".pkl")
                    torch.save(model.state_dict(), model_path)
