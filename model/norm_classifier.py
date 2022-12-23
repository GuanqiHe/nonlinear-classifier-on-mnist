import torch
import torch.nn as nn
import torch.nn.functional as F


class PowClassifier(nn.Module):
    def __init__(self,input_dim, feature_dim,output_dim, norm=2, C=1e-2):
        super(PowClassifier,self).__init__()
        self.feature_dim = feature_dim
        self.norm = norm
        self.h = nn.Parameter(torch.randn(feature_dim, input_dim, requires_grad=True))
        # self.gamma = nn.Parameter(torch.tensor([[1/input_dim] for i in range(feature_dim)], requires_grad=True))
        self.w = nn.Parameter(torch.randn(output_dim, feature_dim, requires_grad=True))
        self.b = nn.Parameter(torch.zeros(output_dim, requires_grad=True))
        # self.param = [self.h, self.gamma, self.w, self.b]
        self.param = [self.h, self.w, self.b]

        torch.nn.init.kaiming_normal_(self.w)

        self.C = C

    def forward(self,x):
        # x = (x - x.mean())/x.sum()
        temp = []
        for i in range(self.feature_dim):
            # print((x - self.h[i]).shape)
            dist = (x - self.h[i]).norm(self.norm, dim=1)
            # print((x - self.h[i]).abs().pow(self.norm))
            # print(res)
            temp.append(dist)
        # print(temp)
        temp = torch.stack(temp)
        # print(temp)
        x = temp.t()@self.w.t() + self.b
        return x
    
    def L1_term(self):
        return torch.Tensor([i.abs().sum() for i in self.param]).sum()*self.C

    def L2_term(self):
        return torch.Tensor([i.square().sum() for i in self.param]).sum()*self.C


if __name__ == "__main__":
    model = PowClassifier(28*28, 20, 10)
    x = torch.randn(4,28*28)
    res = model.forward(x)
    res.sum().backward()
    print(res.shape)
    print(model.h.grad)
    print(model.L1_term())
    print(model.L2_term())