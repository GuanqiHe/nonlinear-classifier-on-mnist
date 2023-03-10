import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms

import os
import datetime

from model.classifier import Classifier


#MNIST 数据集
#设置训练的批次大小、学习率、及训练代数
batch_size=200
learning_rate=0.001
epochs=10

#下载数据集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


model = Classifier(28*28, 10)

#定义优化器，采用SGD随机梯度下降的方式对w1, b1, w2, b2, w3, b3进行优化
# optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
optimizer = optim.SGD(model.param, lr=learning_rate, weight_decay=1e-3)
#定义采用交叉熵作为损失函数
criteon = nn.CrossEntropyLoss()


# 保存模型参数到路径"./data/model_parameter.pkl"
log_dir = "./log/"+"linear_svm_"+str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace(".","_").replace("-","_")+"/"
os.mkdir(log_dir)


# 设置迭代次数
for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据打平为（批次，高度*宽度），-1代表所有
        data = data.view(-1, 28 * 28)

        # 将数据输入到网络中
        cal_data = model.forward(data)
        # 将计算的数据与目标数据求误差损失
        loss = criteon(cal_data, target)

        # 将梯度值初始化为0
        optimizer.zero_grad()
        # pytorch计算梯度值
        loss.backward()
        # 更新梯度值
        optimizer.step()
        # 每隔25*batcsize(200) = 5000 打印输出结果
        if batch_idx % 25 == 0:
            print('训练代数: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    # 将测试误差及正确率清0
    test_loss = 0
    correct = 0
    # 取测试集数据及目标数据
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = model.forward(data)
        # 误差累加
        test_loss += criteon(logits, target).item()
        # 取出预测最大值的索引编号，即预测值
        pred = logits.data.argmax(dim=1)
        # 统计正确预测的个数
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    # 打印输出测试误差及准确率
    print('\n测试集: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

    print(model.state_dict().keys())                                # 输出模型参数名称
    file_dir = log_dir + str(epoch)
    os.mkdir(file_dir)
    torch.save(model.state_dict(), file_dir+"/model_parameter.pkl")
    with open(file_dir+"/log.txt","w") as f:
        f.writelines(['\ntest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))])




    # new_model = classifier()                                                    # 调用模型Model
    # new_model.load_state_dict(torch.load("./data/model_parameter.pkl"))    # 加载模型参数     
    # new_model.forward(input)                                               # 进行

