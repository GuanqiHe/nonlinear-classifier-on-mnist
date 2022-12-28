import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils import data as Data

import os
import datetime

from model.classifier import Classifier
from train import *

load_weights = False
log_dir = "./log/linear_L1_C_1e_2_2022_12_22_17_55_25_700325/"
file_dir = log_dir + "28/"

if __name__=='__main__':




    #MNIST 数据集
    #设置训练的批次大小、学习率、及训练代数
    batch_size=2000
    learning_rate=0.005
    epochs=100


    model = Classifier(28*28, 10, 1e-2)

    if load_weights:
        model.load_state_dict(torch.load(file_dir+"model_parameter.pkl"))    # 加载模型参数   

    optimizer_list = [optim.SGD(model.param, lr=learning_rate)]
    criteon = nn.CrossEntropyLoss()


    # 保存模型参数到路径"./data/model_parameter.pkl"
    log_dir = "./log/"+"linear_L1_C_1e_2_"+str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace(".","_").replace("-","_")+"/"
    os.mkdir(log_dir)

    train(model, optimizer_list, criteon, log_dir, batch_size, epochs)
    
