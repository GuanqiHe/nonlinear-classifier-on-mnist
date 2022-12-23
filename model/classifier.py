import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self,input_dim, output_dim, C=1e-2):
        super(Classifier,self).__init__()

        self.w = nn.Parameter(torch.randn(output_dim, input_dim, requires_grad=True))
        self.b = nn.Parameter(torch.zeros(output_dim, requires_grad=True))
        self.param = [self.w, self.b]

        torch.nn.init.kaiming_normal_(self.w)

        self.C = C

    def forward(self,x):
        x = x@self.w.t() + self.b
        return x
    
    def L1_term(self):
        return (self.w.abs().sum()+self.b.abs().sum())*self.C
    

