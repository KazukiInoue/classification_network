import torch


# torch.cuda.manual_seed_all(8)

a = torch.range(0, 14, 1)

for i in range(3):
    torch.manual_seed(7)
    r = torch.randperm(14)
    b = a[r]
    print(b)


# import numpy as np
#
#
# for i in range(3):
#     np.random.seed(seed=0)
#     a = np.arange(0, 10, dtype=np.int32)
#     np.random.shuffle(a)
#     print(a)