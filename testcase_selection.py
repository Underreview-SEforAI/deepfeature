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

import torchattacks
import time


if __name__ == '__main__':
    
    batch_size = 100

    # test_dataset=TensorDataset(x_test,y_test)
    test_dataset = FashionMNIST(root='../data', train=False, transform=ToTensor(), download=True)
    # test_dataset = SVHN(root='../data', split='test', transform=ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


    dataset = 'fashion'
    model = 'resnet20'
    testmodel = torch.load('./pretrained/'+dataset+'_'+model+'_0.861.pt')
    VS_idx = np.load('./deepfeature/rq4/'+ dataset + '-' + model +'-VS-25.npy')[-5:]
    print(VS_idx)
    # torch.save(testmodel.state_dict(), 'checkpoint/mnist_lenet5_0.918.pt')
    
    # testmodel = ResNet20().cuda()
    # testmodel.load_state_dict(torch.load('./checkpoint/cifar10_epoch_200.pth'), strict=True)

    # attack = PGD(testmodel, eps=2/255, alpha=2/255, steps=1, random_start=True)
    attack = FGSM(testmodel, eps=1/255)
    benign_params = {
                    'shift_x':(0.05,0.15),
                    'shift_y':(0.05,0.15),
                    'rotate':(5,25),
                    'scale':(0.8,1.2),
                    'shear':(15,30),
                    'contrast':(0.5,1.5),
                    'brightness':(0.5,1.5),
                    'blur_mode':'easy'}
    benign_aug = benign_aug(params=benign_params, model = testmodel, seed = 123)


    all_correct_num = 0
    all_sample_num = 0
    testmodel.eval()


    correct_imgs = []
    correct_labels = []

    acc_array = []
    per_feature_acc = []

    batch_neuron_idx = []
    # neuron_idx = np.load('./mnist-lenet1-sensNeuronIdx.npy')
    # neuron_idx = np.array([0])
    neuron_idx = np.load('./'+ dataset + '-' + model +'-sensNeurons.npy')
    sampled_neurons_num = int(neuron_idx.shape[0])
    # sampled_neurons_num = 20
    print(sampled_neurons_num)
    neuron_idx = neuron_idx[-sampled_neurons_num:]
    # print(random_idx)
    # random_idx = [11474, 3489, 12733, 13612, 16289]
    # print(random_idx)
    total_diff = 0
    total_diff_random = 0
    # most_sensitive_samples = []
    # most_sensitive_samples_label = []
    neuron_diff_perSample = []
    neuron_diff_perSample_random = []
    all_adv_samples = []
    all_labels = []
    total_neurons = 0
    start = time.time()



    with tqdm(test_loader) as loader:
        for idx, (test_x, test_label) in enumerate(loader):

            test_x = test_x.cuda()

            predict_y_adv, clean_features = testmodel(test_x)

            # test_x_adv = attack.forward(test_x, test_label)
            test_x_adv = benign_aug(test_x)


            predict_y_adv, adv_features = testmodel(test_x_adv)

            neuron_diff = F.mse_loss(adv_features, clean_features, reduction='none').flatten(start_dim=2)
            # print(neuron_diff.shape)
            # print(clean_features.flatten(start_dim=1).max(dim=1))
            neuron_num = neuron_diff.shape[1]
            # print(neuron_diff.shape)
            # neuron_diff /= clean_features.flatten(start_dim=1).max(dim=1)[0].unsqueeze(1).repeat(1,neuron_num)

            # neuron_diff /= torch.norm(clean_features.flatten(start_dim=1), p=1, dim=1, keepdim=True)
            # neuron_diff /= clean_features
            neuron_diff = neuron_diff.detach().cpu().numpy()
            # print(total_neurons)
            neuron_diff_perSample.append(neuron_diff.mean(axis=2)[:,VS_idx].mean(axis=1))
            # neuron_diff_perSample.append(neuron_diff[:, neuron_idx].mean(axis=1))
            # neuron_diff_perSample_random.append(neuron_diff[:, \
            # np.random.choice(np.linspace(0,neuron_num-1,neuron_num, dtype=np.int32), 20)].mean(axis=1))


            all_adv_samples.append(test_x_adv.cpu().numpy())
            all_labels.append(test_label.cpu().numpy())
            # print(test_label)
            predict_y_adv = np.argmax(predict_y_adv.cpu().detach(), axis=-1)
            current_correct_num = predict_y_adv == test_label
            # print(current_correct_num.shape)
            all_correct_num = current_correct_num.sum()
            all_sample_num = current_correct_num.shape[0]
            acc = all_correct_num / all_sample_num
            # print(all_correct_num)
            acc_array.append(acc)


    # print(np.array(neuron_diff_perSample).shape)
    # print('Sensitivity:', np.array(neuron_diff_perSample).flatten().mean())
    # print('Sensitivity random:', np.array(neuron_diff_perSample_random).flatten().mean())
    K_section = [int(x) for x in np.linspace(0, 10000-1, 2000)]
    sensitive_sample_idx = np.argsort(np.array(neuron_diff_perSample).flatten())[K_section]
    # sensitive_sample_idx = np.random.choice(np.linspace(0,25999, 26000, dtype=np.int32), 5200)
    # random_sample_idx = np.argsort(np.array(neuron_diff_perSample_random).flatten())[-2000:]
    # print(sensitive_sample_idx)
    all_adv_samples = np.array(all_adv_samples)
    all_labels = np.array(all_labels)

    b, bs, c, w, h = all_adv_samples.shape
    most_sensitive_samples = all_adv_samples.reshape(b*bs, c, w, h)[sensitive_sample_idx]
    most_sensitive_samples_label = all_labels.reshape(b * bs)[sensitive_sample_idx]
    end = time.time()
    print(end - start)
    print(most_sensitive_samples.shape)
    print(most_sensitive_samples_label.shape)
    # np.save('./deepfeature/neuron_sensitive_samples_' + dataset + '_' + model + '_images-5.npy', most_sensitive_samples)
    # np.save('./deepfeature/neuron_sensitive_samples_' + dataset + '_' + model + '_labels-5.npy', most_sensitive_samples_label)
    np.save('./deepfeature/rq5/VS_' + dataset + '_' + model + '_images-5_kselection.npy', most_sensitive_samples)
    np.save('./deepfeature/rq5/VS_' + dataset + '_' + model + '_labels-5_kselection.npy', most_sensitive_samples_label)

    # print(total_diff.item() / 256)
    # print(total_diff_random.item() / 256)
    print('Attack Success Rate: ', 1 - np.array(acc_array).mean())
    print('Acc: ', np.array(acc_array).mean())
    print('Total Correct Num: ', np.array(acc_array).sum() * batch_size)

