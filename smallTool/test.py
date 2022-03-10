from random import *
import torch

# batch = [2,4,7,9,27,34]
# cand_mask_pos = [i for i, token in enumerate(batch)]
#
# print("cand_mask_pos:" + str(cand_mask_pos))
#
# shuffle(cand_mask_pos)
#
# print("cand_mask_pos:" + str(cand_mask_pos))
#
# for pos in cand_mask_pos[:2]:
#     print("pos:" + str(pos))
#
# band = []
# small_object = {1:"a"}
# band.append(small_object)
# print('band:' + str(band))
# print(str(band[0].get(1)))
#
# def wayTry():
#     a_id = [1, 2, 4]
#     b_id = [1, 4, 6]
#     c_id = [2, 4, 5]
#
#     return {"a_id": a_id,
#             "b_id": b_id,
#             "c_id": c_id}
#
# a, b, c = tuple(zip(wayTry()))
# d = zip(wayTry())
#
# print(str(a))
# print(str(b))
# print(str(c))
# print(str(d))
# print(list(zip({1:'s'}, "h")))
#
# def wayTry():
#     return "a", "b"
# a, b, c = wayTry()
# print(a, b, c)
a = [1,2,3,4,5,6]
b = torch.tensor(a)
print(b)


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)

node_rep = model(data)
pred_node = node_rep[data.masked_atom_indices]
acc_node = compute_accuracy(pred_node, data.mask_node_label[:, 0])

def accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()