# import torch
from torch import nn
import torch.nn.functional as F
from layer import Linear


MNIST_INPUT_SHAPE = 28 * 28
MNIST_OUTPUT_SHAPE = 10


# Define a simple 2 layer Network
class Net(nn.Module):
    def __init__(self): #, threshold):
        super(Net, self).__init__()
        self.fc1 = Linear(MNIST_INPUT_SHAPE, 300)
        self.fc2 = Linear(300,  100)
        self.fc3 = Linear(100,  MNIST_OUTPUT_SHAPE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


# Define Loss Function -- SGVLB
class Loss(nn.Module):
    def __init__(self, net, train_size):
        super(Loss, self).__init__()
        self.train_size = train_size
        self.net = net

    def forward(self, output, target):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        return F.cross_entropy(output, target) * self.train_size + kl
