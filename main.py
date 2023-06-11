import torch
import Net
import train
import numpy as np
from torch import nn
from Data_Loader import *

"""
预设定参数
"""

train_size = [320,480]
test_size = [512,512]
num_epochs = 20
lr = 1e-4
LR_iter, HR_iter = load_BSDS300_data()
test_LR_iter, test_HR_iter = load_SET11_modified_data()
loss = nn.MSELoss(reduction="none")
# device = torch.device("cpu")
# device = torch.device("mps")
device = torch.device("cuda")
train_way = "pre"

"""
模型设置
"""

net = Net.SRCNN()
# net = Net.VDSR()
# net = Net.DRRN()
# net = Net.FSRCNN(4)
# net = Net.LapSRN()

"""
训练设置
"""

# train_acc, train_loss, test_acc = net.train(
#     train_iter, test_iter, num_epochs, lr, "default"
# )

net.to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas = (0.9, 0.999))
train_loss, test_PSNR= train.train(
    net, LR_iter, HR_iter, train_size, test_LR_iter, test_HR_iter, test_size, num_epochs, loss, optimizer, device, train_way
)
np.savetxt('./XZ3/train_loss.txt',train_loss)
np.savetxt('./XZ3/test_PSNR.txt',test_PSNR)
train.evaluate_res(net, test_LR_iter, test_size, device, train_way)