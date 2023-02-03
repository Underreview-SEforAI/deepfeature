import numpy as np
import os
import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from feature_wise_pgd import PGD
import matplotlib.pyplot as plt
from resnet20 import ResNet20

if __name__ == '__main__':
    batch_size = 100
    # test_dataset = CIFAR10(root='../data', train=False, transform=ToTensor(), download=True)
    test_dataset = SVHN(root='../data', split='test', transform=ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    device = torch.device("cuda")
    # model = ResNet20(num_classes=10).to(device)
    # model.load_state_dict(torch.load('./pretrained/cifar10_lenet1_0.861.pt'), strict=True)
    model = torch.load('./pretrained/svhn_lenet5_0.842.pt')
    # model = torch.load('./pretrained/cifar10_vgg16_0.825.pt')
    model = model.to(device)
    feature_wise_attack = PGD(model = model, eps = 0.15, alpha= 0.05, steps = 20)

    all_correct_num = 0
    all_sample_num = 0
    model.eval()

    correct = 0
    acc_array_epoch = []
    correct_array_epoch = []
    for idx, (test_x, test_label) in enumerate(tqdm(test_loader)):
        test_x = test_x.to(device)
        predict,_ = model(test_x)
        predict = np.argmax(predict.cpu().detach(), axis=-1)
        current_correct = predict == test_label
        correct += np.sum(current_correct.numpy(), axis=-1)
        adv_images_array, clean_feature_map, adv_feature_map_array = feature_wise_attack.forward(test_x, test_label.to(device))
        # print(torch.tensor([item.cpu().detach().numpy() for item in adv_feature_map_array] ).shape)
        acc_array_channel = []
        correct_num_array = []
        for adv_images in adv_images_array:
            predict_y, _ = model(adv_images)
            predict_y = np.argmax(predict_y.cpu().detach(), axis=-1)
            # print(predict_y.shape)
            current_correct_num = predict_y == test_label
            all_correct_num = np.sum(current_correct_num.numpy(), axis=-1)
            correct_num_array.append(all_correct_num)
            # all_sample_num = current_correct_num.shape[0]
            # acc = 100- (all_correct_num / all_sample_num * 100)
            # acc_array_channel.append(acc)




        correct_array_epoch.append(correct_num_array)


    sorted_feature_wise_acc = torch.from_numpy(np.argsort(np.mean(np.array(acc_array_epoch), axis=0)))
    print(sorted_feature_wise_acc)
    # print(torch.from_numpy(np.mean(np.array(acc_array_epoch), axis=0)))
    attack_success_rate = [] 
    print(np.sum(np.array(correct_array_epoch), axis=0))
    correct_array_epoch = np.sum(np.array(correct_array_epoch), axis=0)
    for correct_num in correct_array_epoch:
        attack_success_rate.append(100- correct_num/correct*100)
    print(attack_success_rate)
    print(np.argsort(attack_success_rate))