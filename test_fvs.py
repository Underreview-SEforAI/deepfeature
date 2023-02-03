import numpy as np
import os
import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor
# from feature_wise_pgd import PGD
import random
from PGD import PGD, FGSM
# from mcmc import MCMC
from resnet20 import ResNet20
from LeNet import LeNet5
from benign_perturbations import benign_aug
from feature_map_plot import feature_map_plot

import torchattacks



if __name__ == '__main__':
    
    batch_size = 100

    # test_dataset=TensorDataset(x_test,y_test)
    test_dataset = FashionMNIST(root='../data', train=False, transform=ToTensor(), download=True)
    # test_dataset = SVHN(root='../data', split='test', transform=ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    testmodel = torch.load('./pretrained/fashion_lenet1_0.852.pt')
    attack = PGD(testmodel, eps=0.1, alpha=0.03, steps=7)
    benign_params = {
                    'shift_x':(0.05,0.15),
                    'shift_y':(0.05,0.15),
                    'rotate':(5,25),
                    'scale':(0.8,1.2),
                    'shear':(15,30),
                    'contrast':(0.5,1.5),
                    'brightness':(0.5,1.5),
                    'blur_mode':'easy'}
    benign_aug = benign_aug(params=benign_params, model = testmodel, seed = 0)


    all_correct_num = 0
    all_sample_num = 0
    testmodel.eval()


    correct_imgs = []
    correct_labels = []

    acc_array = []
    per_feature_acc = []

    batch_neuron_idx = []
    neuron_dict = {}
    total_diff = 0
    total_diff_random = 0
    neuron_diff_perSample = []
    neuron_diff_perSample_random = []
    all_adv_samples = []
    all_labels = []
    total_neurons = 0
    avg_feature_diff = []

    with tqdm(test_loader) as loader:
        for idx, (test_x, test_label) in enumerate(loader):
            # if idx == 100:
                # break
            test_x = test_x.cuda()

            predict_y_adv, clean_features = testmodel(test_x)

            # test_x_adv = attack.forward(test_x, test_label)
            test_x_adv = benign_aug(test_x)


            predict_y_adv, adv_features = testmodel(test_x_adv)

            # print(F.mse_loss(adv_features, clean_features, reduction='none').flatten(start_dim=2).shape)
            neuron_diff = F.mse_loss(adv_features, clean_features, reduction='none').flatten(start_dim=2) #/ clean_features.flatten(start_dim=2).max(dim=2)[0].unsqueeze(2)
            # neuron_diff = ((1+adv_features)/(1+clean_features)).flatten(start_dim=2)
            avg_feature_diff.append(neuron_diff.mean(dim=2).mean(dim=0).detach().cpu().numpy())
            # print(neuron_diff.mean(dim=2).mean(dim=0).shape)
            # print(clean_features.flatten(start_dim=1).max(dim=1))
            neuron_num = neuron_diff.shape[1]
            selected_neurons_num = 1000 if neuron_num > 1000 else neuron_num
            batch_neuron_idx += torch.argsort(neuron_diff)[:,-selected_neurons_num:].flatten().cpu()
            # print(torch.argsort(max_diff_neurons)[-5:])


            # print(test_label)
            predict_y_adv = np.argmax(predict_y_adv.cpu().detach(), axis=-1)
            current_correct_num = predict_y_adv == test_label
            # print(current_correct_num.shape)
            all_correct_num = current_correct_num.sum()
            all_sample_num = current_correct_num.shape[0]
            acc = all_correct_num / all_sample_num
            # print(all_correct_num)
            acc_array.append(acc)

    feature_fdl = np.mean(np.array(avg_feature_diff),axis=0)
    print(list(feature_fdl))
    # # print(per_feature_acc)
    # batch_neuron_idx += torch.linspace(0, total_neurons-1, total_neurons, dtype=torch.int)
    # print(np.array(avg_feature_diff, dtype=np.float32).shape)
    print(np.argsort(np.mean(np.array(avg_feature_diff),axis=0)))

