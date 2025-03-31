import os
import random
import numpy as np
import torch
from src.mkdpinn import MKDPINN
from src.preprocessing import get_data_loader
import torch.nn as nn




if __name__ == '__main__':

    sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    # windows length
    sequence_length = 15
    # smoothing intensity
    alpha = 0.1
    # max RUL
    threshold = 125
    batch_size = 256
    dataset = 'FD004'
    train_loader, valid_loader, test_loader = get_data_loader(dataset=dataset,
                                                              sensors=sensors, sequence_length=sequence_length,
                                                              alpha=alpha, threshold=125,
                                                              batch_size=batch_size, train_size=0.9)
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    meta_param = [0.001,8,0.1,50,64,5,0]
    mkdpinn = MKDPINN(hidden_dim=64, derivatives_order=2, meta_params=meta_param)
    # mkdpinn.train_model(train_loader, valid_loader,test_loader,resume_from_checkpoint=False)
    mkdpinn.test(test_loader,MSE = nn.MSELoss(),in_train=False,dataset=dataset)
