import torch
import torch.optim as optim

import os
import datetime

from model.classifier import Classifier
from model.torch_rbf import *
from train import *
import cv2
import numpy as np


load_weights = True
log_dir = "./log/pos1_norm2_2022_12_23_17_31_44_928448/"
file_dir = log_dir + "8/"



class Network(nn.Module):
    
    def __init__(self, layer_widths, layer_centres, basis_func, centres_init=None):
        super(Network, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(RBF(layer_widths[i], layer_centres[i], basis_func, centres_init))
            self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i+1]))
    
    def forward(self, x):
        out = (x - x.mean())/(x.var().sqrt())
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out
    
    def L1_term(self):
        return torch.Tensor([param.abs().sum() for param in self.parameters()]).sum()*1e-2


if __name__=='__main__':

    
    initial_centres = torch.Tensor(np.array([cv2.imread("./data/"+str(i)+".png", flags=cv2.IMREAD_GRAYSCALE) for i in range(10)])).reshape(10, 784)
    initial_centres = torch.concatenate([(initial_centres-initial_centres.mean())/((initial_centres.var()).sqrt()), torch.randn(10, 784)], dim=0)

    #MNIST 数据集
    #设置训练的批次大小、学习率、及训练代数
    batch_size=1000
    learning_rate=5e-3
    epochs=100

    layer_widths = [28*28, 10]
    layer_centres = [20]
    basis_func = poisson_one

    model = Network(layer_widths, layer_centres, basis_func, initial_centres)
    # model = Classifier(28*28, 10)

    optimizer_list = [torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)]

    if load_weights:
        model.load_state_dict(torch.load(file_dir+"model_parameter.pkl"))    # 加载模型参数   

    


    # 保存模型参数到路径"./data/model_parameter.pkl"
    log_dir = "./log/"+"pos1_norm2_"+str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace(".","_").replace("-","_")+"/"
    os.mkdir(log_dir)

    criteon = nn.CrossEntropyLoss()

    train(model,optimizer_list, criteon ,log_dir, batch_size, epochs)