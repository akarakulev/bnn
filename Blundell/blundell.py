import math
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Parameter


torch.manual_seed(0)


def factorized_normal(x, mu, sigma):
    likelihood = torch.exp(-0.5 * (x - mu)**2 / sigma**2) / np.sqrt(2.0*np.pi) / sigma
    return torch.clamp(likelihood, 1e-10, 1.)  # to avoid numerical issues


def log_factorized_normal(x, mu, sigma):
    log_likelihood = -0.5 * (x - mu)**2 / sigma**2 - sigma
    return torch.clamp(log_likelihood, np.log(1e-10), 0.)


def log_mixture_prior(input_, weight_first=0.25, sigma_first=1., sigma_second=np.exp(-7)):
    sigma_first = torch.FloatTensor([sigma_first])
    sigma_second = torch.FloatTensor([sigma_second])
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
        self.reset_parameters()

        self.kl_term = 0.

    def reset_parameters(self):
        self.W.data.normal_(0., .1)
        # self.W_rho.data.fill_(-3.)
        self.W_rho.data.uniform_(-3., -3.)
        self.bias.data.normal_(0., .1)
        # self.bias_rho.data.fill_(-3.)
        self.bias_rho.data.uniform_(-3., -3.)

    def forward(self, x):

        if self.training:
            sigma = torch.log(1. + torch.exp(self.W_rho))
            bias_sigma = torch.log(1. + torch.exp(self.bias_rho))

            epsilon_weight = sigma.data.new(sigma.size()).normal_()
            epsilon_bias = bias_sigma.data.new(bias_sigma.size()).normal_()
            W_sample = self.W + sigma * epsilon_weight
            bias_sample = self.bias + bias_sigma * epsilon_bias

            log_prior = log_mixture_prior(W_sample).sum() + log_mixture_prior(bias_sample).sum()
            log_post = (
                log_factorized_normal(W_sample, self.W, sigma).sum()
                + log_factorized_normal(bias_sample, self.bias, bias_sigma).sum()
            )
            self.kl_term = log_post - log_prior
            return F.linear(x, W_sample) + bias_sample

        return F.linear(x, self.W) + self.bias

    def kl_reg(self):
        return self.kl_term


class Loss(nn.Module):
    def __init__(self, net, train_size, num_samples=5, problem='regression'):
        super().__init__()
        self.train_size = train_size
        self.net = net
        self.num_samples = num_samples
        self.problem = problem

    def get_kl(self):
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
                kl += module.kl_reg()
        return kl

    def forward(self, input_, target):
        assert not target.requires_grad
        
        kl, log_likelihood = 0., 0.
        for _ in range(self.num_samples):
            output = self.net.forward(input_)
            sample_kl = self.get_kl()
            if self.problem == 'classification':
                sample_log_likelihood = -F.nll_loss(output, target, reduction='sum')
            elif self.problem == 'regression':
                sample_log_likelihood = -(.5 * (target - output)**2).sum()

            kl += sample_kl
            log_likelihood += sample_log_likelihood

        kl /= self.num_samples
        log_likelihood /= self.num_samples
        scaler = self.train_size / target.size(0)
        return kl - log_likelihood * scaler

