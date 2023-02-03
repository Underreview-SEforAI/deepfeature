import numpy as np
import os
import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR10, MNIST, SVHN
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

import torchattacks



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('-p', '--percentage', type=int, default=20, help='Top k percentage of selected cases')
    args = parser.parse_args()

    dataset = 'svhn'
    model = 'vgg16'
    # Dataset
    eval_images_path = './deepfeature/rq5/VS_' + dataset + '_' + model + '_images-5_kselection.npy'
    eval_labels_path = './deepfeature/rq5/VS_' + dataset + '_' + model + '_labels-5_kselection.npy'

    test_x_numpy = np.load(eval_images_path)
    test_y_numpy = np.load(eval_labels_path)
    # print(test_x_numpy.shape, test_y_numpy.shape)58.2
    total_samples_num = test_y_numpy.shape[0]
    x_test=torch.from_numpy(test_x_numpy)
    y_test=torch.from_numpy(test_y_numpy)
    x_test = torch.flip(x_test, dims=[0])
    y_test = torch.flip(y_test, dims=[0])
    test_dataset=TensorDataset(x_test,y_test)
    # test_dataset = CIFAR10(root='../data', train=False, transform=ToTensor(), download=True)

    batch_size = 100
    test_dataset = SVHN(root='../data', split='test', transform=ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    


    # Model Definition & Checkpoint reload
    testmodel = torch.load('./pretrained/svhn_vgg16_0.920.pt').cuda()
    # testmodel = ResNet20().cuda()
    # testmodel.load_state_dict(torch.load('./checkpoint/cifar10_epoch_200.pth'), strict=True)
    # testmodel.cuda()

    batch_idx = int((total_samples_num * args.percentage) / (100 * batch_size))
    print(total_samples_num,batch_idx)
    acc_array = []

    activated_count = np.zeros(100)

    all_correct_num = 0
    all_sample_num = 0
    with tqdm(test_loader) as loader:
        for idx, (test_x, test_label) in enumerate(loader):
            # if idx == args.percentage * 100 / batch_size:
            #     break
            test_x, test_label = test_x.cuda(), test_label.cuda()

            predict_y_adv, features = testmodel(test_x)
            print(features.shape)
            predict_y_adv = np.argmax(predict_y_adv.cpu().detach(), axis=-1)
            current_correct_num = predict_y_adv == test_label.cpu()


            # print(current_correct_num.shape)
            all_correct_num += current_correct_num.sum().item()
            # print(all_correct_num)
            all_sample_num += test_label.shape[0]
            acc = current_correct_num.sum().item() / test_x.shape[0]

            acc_array.append(acc)

    print(1 - all_correct_num / all_sample_num)
    print('Attack Success Rate: ', 1 - np.mean(acc_array))
    print('Acc: ', np.mean(acc_array))
    print('Total Correct Num: ', np.sum(acc_array) * batch_size)
