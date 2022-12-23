import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils import data as Data
import os
import re


def train(model, optmizer_list, criteon ,log_dir, batch_size=2500, epochs=100):
    #下载数据集
    train_loader = Data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True, num_workers=6)
    test_loader = Data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, num_workers=2)

    if torch.cuda.is_available():
        model = model.cuda()
        criteon = criteon.cuda()


    # 设置迭代次数
    for epoch in range(epochs):
        log = []

        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据打平为（批次，高度*宽度），-1代表所有

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            data = data.view(-1, 28 * 28)

            for optimizer in optmizer_list: 
                optimizer.zero_grad()

            # 将数据输入到网络中
            cal_data = model.forward(data)
            # 将计算的数据与目标数据求误差损失
            data_loss = criteon(cal_data, target)
            loss =  data_loss + model.L1_term()

            # 将梯度值初始化为0
            # pytorch计算梯度值
            loss.backward()
            # 更新梯度值
            for optimizer in optmizer_list:
                optimizer.step()
            # 每隔25*batcsize(200) = 5000 打印输出结果
            if batch_idx % int(5000/batch_size) == 0:
                log.append('train: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, L1 term: {:.6f}, learning_rate: {} \n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), data_loss.item(), model.L1_term().item(), str([opt.state_dict()['param_groups'][0]['lr'] for opt in optmizer_list])))
                print(log[-1])
                # print("Grad: centres: {:.12f}, sigm: {:.6f},".format(model.rbf_layers[0].centres.grad.mean(), model.rbf_layers[0].log_sigmas.grad.mean()))
                # print([i.max() for i in model.param])
                # print([i.min() for i in model.param])
        

        test_loss, correct = eval(model, test_loader, criteon)

        file_dir = log_dir + str(epoch)
        os.mkdir(file_dir)
        save_model(model, file_dir+"/model_parameter.pkl" )
        gen_log(file_dir+"/log.txt", log, loss, test_loss, correct, len(test_loader.dataset))
        



def eval(model, test_loader, criteon):
    # 将测试误差及正确率清0
    test_loss = 0
    correct = 0
    # 取测试集数据及目标数据
    for data, target in test_loader:

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss, correct




def save_model(model,file_path):
    print("model parameters: ",model.state_dict().keys())                                # 输出模型参数名称
    torch.save(model.state_dict(), file_path)




def gen_log(file_path, log, train_loss, test_loss, correct, data_len):
    with open(file_path,"w") as f:
        f.writelines(log+ ['\nTest set: Total loss: [{:.8f}] Average loss: [{:.8f}], Accuracy: {}/{} [{:.4f}%]\n'.format(
        train_loss.item(), test_loss, correct, data_len,
        100. * correct / data_len)])


def load_log(log_dir):
    folder_name = os.listdir(log_dir)
    folder_name.sort(key=lambda x: int(x))
    log_name = "log.txt"

    losses = []
    accuracies = []
    epochs = []

    for name in folder_name:
        log_path = os.path.join(log_dir,name,log_name)
        with open(log_path, "r") as f:
            lines = f.readlines()
            evaluate = lines[-1]
            res = re.findall("\\[.*?\\]",evaluate)
            data = [i[1:-1] for i in res]
            data[-1] = data[-1][:-1]
            data = [float(i) for i in data]
            loss, _, accuracy =  data
            losses.append(loss)
            accuracies.append(accuracy)
            epochs.append(float(name))

    return epochs, losses, accuracies  