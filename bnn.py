import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_sigma.data.fill_(-5)

    def forward(self, x):

        if self.training:
            lrt_mean =  F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps

        return F.linear(x, self.W) + self.bias

    def kl_reg(self):
        # Return KL here -- a scalar
        kl = 0.5 * (torch.norm(self.W)**2 + torch.norm(self.bias)**2)
        return kl


# Define Loss Function -- SGVLB
class Loss(nn.Module):
    def __init__(self, net, train_size, problem='regression'):
        super(Loss, self).__init__()
        self.train_size = train_size
        self.net = net
        self.problem = problem

    def forward(self, output, target):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        if self.problem == 'regression':
            neg_log_likelihood = -(.5 * (target - output)**2).sum() / target.size(0)
        elif self.problem == 'classification':
            neg_log_likelihood = F.cross_entropy(output, target)
        
        return neg_log_likelihood * self.train_size + kl
