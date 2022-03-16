import apex
import numpy as np
import torch
a = np.array([1.0000,1.0000,1.0000])
print(np.sum(a))

x = torch.tensor([10.0])
x = x.cuda()
print(x)