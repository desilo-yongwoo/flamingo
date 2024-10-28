import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from util.resnet import ResNet18

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.resnet = ResNet18()

    def forward(self, x):
        x = self.resnet(x)
        return x
