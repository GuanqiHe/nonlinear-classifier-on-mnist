import torch
import torch.nn as nn

class RBFClassifier(nn.Module):
    
    def __init__(self, layer_widths, layer_centres, basis_func, dist_func, centres_init=None, L1_weight=0.0):
        super(RBFClassifier, self).__init__()
        self.L1_weight = L1_weight
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(RBF(layer_widths[i], layer_centres[i], basis_func, dist_func, centres_init, L1_weight))
            self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i+1]))
    
    def forward(self, x):
        out = (x - x.mean())/(x.var().sqrt())
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out
    
    def L1_term(self):
        return torch.Tensor([param.abs().sum() for param in self.parameters()]).sum()*self.L1_weight


# RBF Layer

class RBF(nn.Module):

    def __init__(self, in_features, out_features, basis_func, dist_func, centres_init=None, L1_weight = 0.0):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if centres_init is not None:
            self.centres = nn.Parameter(torch.Tensor(centres_init))
        else:
            self.centres = nn.Parameter(torch.randn(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.zeros(out_features))
        self.basis_func = basis_func
        self.dist_func = dist_func
        self.L1_weight = L1_weight
        if centres_init is None:
            self.reset_parameters()


    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = self.dist_func(x, c)
        distances = distances / torch.exp(self.log_sigmas).unsqueeze(0) / float(self.out_features)
        # print(distances)
        return self.basis_func(distances)
    
    def L1_term(self):
        return torch.Tensor([param.abs().sum() for param in self.parameters()]).sum()*self.L1_weight
    

# RBFs

def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi

def basis_func_dict():
    
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases

def norm_one(x, y):
    return (x - y).abs().sum(-1)

def norm_two(x, y):
    return (x - y).pow(2).sum(-1).pow(0.5)

def basis_dist_dict():

    dists = {
        '1-norm': norm_one,
        '2-norm': norm_two
    }

    return dists