from __future__ import print_function
import argparse
import logging
import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torch.nn import CrossEntropyLoss
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader,TensorDataset
from torchvision.transforms import ToTensor
from resnet20 import ResNet20
from tqdm import tqdm

def train(args, model, device, train_loader, optimizer, epoch):
    """Training"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    for i in range(10):
        with tqdm(train_loader[i]) as loader:
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output,_ = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    



def test(args, model, device, test_loader, epoch):
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,_ = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nEpoch {}  Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('--save_path', type=str, default='./checkpoint', help='saved weight path')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--data_root', type=str, default='../data/', help='dataset path')
    parser.add_argument('--attack', type=str, default=None, help='attack method')
    parser.add_argument('--epoch', type=int, default=36, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--n_classes', type=int, default=10, help='num classes')
    parser.add_argument('-p', '--percentage', type=int, default=10, help='Top k percentage of selected cases')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
                                # transforms.RandomCrop(32, padding=4),
                                # transforms.RandomHorizontalFlip(),  
                                transforms.ToTensor(),])
    
    dataset = 'svhn'
    model = 'vgg16'
    # Dataset
    eval_images_path = './neuron_sensitive_samples_random_' + dataset + '_' + model + '_images.npy'
    eval_labels_path = './neuron_sensitive_samples_random_' + dataset + '_' + model + '_labels.npy'


    test_x_numpy = np.load(eval_images_path)
    test_y_numpy = np.load(eval_labels_path)
    samples_num = int(args.percentage / 20 * test_y_numpy.shape[0])
    print(samples_num)
    x_test=torch.from_numpy(test_x_numpy)
    y_test=torch.from_numpy(test_y_numpy)
    x_test = torch.flip(x_test, dims=[0])[:samples_num]    
    y_test = torch.flip(y_test, dims=[0])[:samples_num].squeeze()
    train_dataset=TensorDataset(x_test,y_test)

    
    train_dataset_ori = SVHN(root='../data', split='train', transform=ToTensor(), download=True)
    test_dataset = SVHN(root='../data', split='test', transform=ToTensor(), download=True)
    train_loader = (DataLoader(train_dataset, batch_size=args.batch_size),\
                    DataLoader(train_dataset_ori, batch_size=args.batch_size))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    

    model = torch.load('./pretrained/svhn_vgg16_0.920.pt').cuda()


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30])
    # Train + Test per epoch
    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        curr_acc = test(args, model, device, test_loader, epoch)
        if curr_acc>best_acc:
            torch.save(model.state_dict(), './retrained/svhn_vgg16_p%d.pt' % (args.percentage))
            best_acc = curr_acc
        scheduler.step()
    print('Best Acc: ', best_acc)
    # Do checkpointing - Is saved in outf
    # torch.save(model.state_dict(), './checkpoint/cifar10_epoch_%d.pth' % (args.epoch))