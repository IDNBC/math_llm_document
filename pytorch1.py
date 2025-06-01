# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import sentencepiece as spm
# import json

import torch
print(torch.cuda.is_available())        # True ならGPU利用可能
print(torch.cuda.get_device_name(0))   # GPU名が出るはず
