import math
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Parameter


torch.manual_seed(0)  # for reproducibility
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


INPUT_SHAPE = 1
OUTPUT_SHAPE = 1
WEIGHT_1 = 0.25
SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)
NUM_SAMPLES = 5

def factorized_normal(x, mu, sigma):
    likelihood = torch.exp(-0.5 * (x - mu)**2 / sigma**2) / np.sqrt(2.0*np.pi) / sigma
    return torch.clamp(likelihood, 1e-10, 1.)  # clip to avoid numerical issues


def log_factorized_normal(x, mu, sigma):
    log_likelihood = -0.5 * (x - mu)**2 / sigma**2 - sigma
    return torch.clamp(log_likelihood, -10, 0.)


def log_mixture_prior(input_, weight_first=WEIGHT_1, sigma_first=SIGMA_1, sigma_second=SIGMA_2):
    prob_first = factorized_normal(input_, 0., sigma_first)
    prob_second = factorized_normal(input_, 0., sigma_second)
    likelihood = weight_first * prob_first + (1 - weight_first) * prob_second
    return torch.log(likelihood)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))
        self.bias_rho = Parameter(torch.Tensor(1, out_features))
        # self.log_sigma = Parameter(torch.Tensor(out_features, in_features))
        # self.log_bias_sigma = Parameter(torch.Tensor(1, out_features))
        self.reset_parameters()

        # Initialise prior and posterior
        self.log_prior = 0.
        self.log_post = 0.

    def reset_parameters(self):
        self.W.data.normal_(0., .1)
        self.W_rho.data.uniform_(-3., -3.)
        self.bias.data.normal_(0., .1)
        self.bias_rho.data.uniform_(-3., -3.)
        # self.log_sigma.data.fill_(-5)
        # self.log_bias_sigma.data.fill_(-5)

    def forward(self, x):

        if self.training:
            sigma = torch.log(1. + torch.exp(self.W_rho))
            bias_sigma = torch.log(1. + torch.exp(self.bias_rho))

            epsilon_weight = sigma.data.new(sigma.size()).normal_()
            epsilon_bias = bias_sigma.data.new(bias_sigma.size()).normal_()
            W_sample = self.W + sigma * epsilon_weight
            bias_sample = self.bias + bias_sigma * epsilon_bias

            # Compute posterior and prior probabilities
            self.log_prior = log_mixture_prior(W_sample).sum() + log_mixture_prior(bias_sample).sum()
            self.log_post = (
                torch.log(factorized_normal(W_sample, self.W, sigma)).sum()
                + torch.log(factorized_normal(bias_sample, self.bias, bias_sigma)).sum()
            )
            return F.linear(x, W_sample) + self.bias

            # lrt_mean =  F.linear(x, self.W) + self.bias
            # lrt_std = torch.sqrt(F.linear(x * x, sigma * sigma) + bias_sigma * bias_sigma + 1e-8)
            # eps = lrt_std.data.new(lrt_std.size()).normal_()
            # return lrt_mean + lrt_std * eps #+ bias_sigma * epsilon_bias

        return F.linear(x, self.W) + self.bias

    def kl_reg(self):
        # Return KL here -- a scalar
        kl = self.log_post - self.log_prior
        return kl


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(INPUT_SHAPE, 16)
        self.fc2 = Linear(16,  16)
        self.fc3 = Linear(16, 16)
        self.fc4 = Linear(16,  OUTPUT_SHAPE)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Define Loss Function -- SGVLB
class Loss(nn.Module):
    def __init__(self, net, train_size):
        super().__init__()
        self.train_size = train_size
        self.net = net
        self.num_samples = NUM_SAMPLES

    def get_kl(self):
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
                kl += module.kl_reg()
        return kl

    def forward(self, input_, target):
        assert not target.requires_grad
        
        # output = self.net.forward(input_)
        # # log_likelihood = -F.cross_entropy(F.log_softmax(output, dim=1), target) * self.train_size
        # log_likelihood = -(.5 * (target - output) ** 2).sum() #* self.train_size
        # kl = self.get_kl()
        # # Note KL weighting
        # return 0.1 * kl - log_likelihood

        kl, log_likelihood = 0., 0.
        for _ in range(self.num_samples):
            output = self.net.forward(input_)
            sample_kl = self.get_kl()
            # if self.CLASSES > 1:
            #     sample_log_likelihood = -F.nll_loss(output, target, reduction='sum')
            # else:
            sample_log_likelihood = -(.5 * (target - output)**2).sum()
            kl += sample_kl
            log_likelihood += sample_log_likelihood

        kl /= self.num_samples
        log_likelihood /= self.num_samples
        return (0.1 * kl - log_likelihood)
        # scaler = self.train_size / target.size(0)
        # return (kl - log_likelihood * scaler) * 0.1

