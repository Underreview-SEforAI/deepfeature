import numpy as np
import os
import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR10, MNIST
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor
# from feature_wise_pgd import PGD
import argparse
import random
from PGD import PGD
# from mcmc import MCMC
from resnet20 import ResNet20
from LeNet import LeNet5
from benign_perturbations import benign_aug
from torchsummary import summary
import torchattacks


if __name__ == '__main__':

    # Model Definition & Checkpoint reload
    model = ResNet20().cuda()
    model.load_state_dict(torch.load('./checkpoint/cifar10_epoch_200.pth'), strict=True) #544,330
    model = torch.load('./pretrained/fashion_resnet20_0.861.pt').cuda()
    summary(model, (1,32,32))