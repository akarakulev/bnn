import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(0)  # for reproducibility
hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}

# GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)
def gaussian(x, mu, sigma):
    # bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    # return torch.clamp(GAUSSIAN_SCALER / sigma * bell, 1e-10, 1.)  # clip to avoid numerical issues
    likelihood = torch.exp(-0.5 * (x - mu)**2 / sigma**2) / np.sqrt(2.0*np.pi) / sigma
    return torch.clamp(likelihood, 1e-10, 1.)  # clip to avoid numerical issues


def scale_mixture_prior(input_, PI, SIGMA_1, SIGMA_2):
    prob1 = PI * gaussian(input_, 0., SIGMA_1)
    prob2 = (1. - PI) * gaussian(input_, 0., SIGMA_2)
    return torch.log(prob1 + prob2)


# Single Bayesian fully connected Layer with linear activation function
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, parent):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialise weights and bias
        if parent.GOOGLE_INIT: # These are used in the Tensorflow implementation.
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0., .05))  # or .01
            self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-5., .05))  # or -4
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0., .05))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-5., .05))
        else: # These are the ones we've been using so far.
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0., .1))
            self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3., -3.))
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0., .1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3., -3.))

        # Initialise prior and posterior
        self.lpw = 0.
        self.lqw = 0.

        self.PI = parent.PI
        self.SIGMA_1 = parent.SIGMA_1
        self.SIGMA_2 = parent.SIGMA_2
        self.hasScalarMixturePrior = parent.hasScalarMixturePrior

    # Forward propagation
    def forward(self, input_, infer=False):
        if infer:
            return F.linear(input_, self.weight_mu, self.bias_mu)

        # Obtain positive sigma from logsigma, as in paper
        weight_sigma = torch.log(1. + torch.exp(self.weight_rho))
        bias_sigma = torch.log(1. + torch.exp(self.bias_rho))

        # Sample weights and bias
        epsilon_weight = Variable(torch.Tensor(self.out_features, self.in_features).normal_(0., 1.)).to(DEVICE)
        epsilon_bias = Variable(torch.Tensor(self.out_features).normal_(0., 1.)).to(DEVICE)
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias

        # Compute posterior and prior probabilities
        self.lpw = scale_mixture_prior(weight, self.PI, self.SIGMA_1, self.SIGMA_2).sum() + scale_mixture_prior(
                bias, self.PI, self.SIGMA_1, self.SIGMA_2).sum()

        self.lqw = torch.log(gaussian(weight, self.weight_mu, weight_sigma)).sum() + torch.log(
            gaussian(bias, self.bias_mu, bias_sigma)).sum()

        # Pass sampled weights and bias on to linear layer
        return F.linear(input_, weight, bias)

    def kl_reg(self):
        return self.lqw - self.lpw
        # return 0.5 * (torch.norm(self.weight_mu)**2 + torch.norm(self.bias_mu)**2)


class BayesianNetwork(nn.Module):
    def __init__(self, inputSize, CLASSES, layers, activations, SAMPLES, BATCH_SIZE, NUM_BATCHES, hasScalarMixturePrior,
                 PI, SIGMA_1, SIGMA_2, GOOGLE_INIT=False):
        super().__init__()
        self.inputSize = inputSize
        self.activations = activations
        self.CLASSES = CLASSES
        self.SAMPLES = SAMPLES
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_BATCHES = NUM_BATCHES
        self.train_size = BATCH_SIZE * NUM_BATCHES
        self.DEPTH = 0  # captures depth of network
        self.GOOGLE_INIT = GOOGLE_INIT
        # to make sure that number of hidden layers is one less than number of activation function
        assert (activations.size - layers.size) == 1

        self.SIGMA_1 = SIGMA_1
        self.hasScalarMixturePrior = hasScalarMixturePrior
        if hasScalarMixturePrior == True:
            self.SIGMA_2 = SIGMA_2
            self.PI = PI

        self.layers = nn.ModuleList([])  # To combine consecutive layers
        if layers.size == 0:
            self.layers.append(BayesianLinear(inputSize, CLASSES, self))
            self.DEPTH += 1
        else:
            self.layers.append(BayesianLinear(inputSize, layers[0], self))
            self.DEPTH += 1
            for i in range(layers.size - 1):
                self.layers.append(BayesianLinear(layers[i], layers[i + 1], self))
                self.DEPTH += 1
            self.layers.append(BayesianLinear(layers[layers.size - 1], CLASSES, self))  # output layer
            self.DEPTH += 1

    # Forward propagation and assigning activation functions to linear layers
    def forward(self, x, infer=False):
        x = x.view(-1, self.inputSize)
        layerNumber = 0
        for i in range(self.activations.size):
            if self.activations[i] == 'relu':
                x = F.relu(self.layers[layerNumber](x, infer))
            elif self.activations[i] == 'softmax':
                x = F.log_softmax(self.layers[layerNumber](x, infer), dim=1)
            else:
                x = self.layers[layerNumber](x, infer)
            layerNumber += 1
        return x

    def get_kl(self):
        kl = 0
        for module in list(self.children())[0]:
            if hasattr(module, 'kl_reg'):
                kl += module.kl_reg()
        return kl

    def BBB_loss(self, input_, target):

        sample_log_likelihood, sample_kl = 0., 0.
        for _ in range(self.SAMPLES):
            output = self.forward(input_)
            sample_kl += self.get_kl()
            if self.CLASSES > 1:
                sample_log_likelihood += -F.nll_loss(output, target, reduction='sum')
            else:
                sample_log_likelihood += -(.5 * (target - output) ** 2).sum()

        l_likelihood = sample_log_likelihood / self.SAMPLES
        kl = sample_kl / self.SAMPLES

        # KL weighting
        return  (kl - l_likelihood * self.train_size / target.size(0)) * 0.1