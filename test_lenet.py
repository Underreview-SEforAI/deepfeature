from LeNet import LeNet_5
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from feature_wise_pgd import PGD
import matplotlib.pyplot as plt

if __name__ == '__main__':
    batch_size = 256
    # train_dataset = mnist.MNIST(root='./mnist_train', train=True, transform=ToTensor(), download=True)
    test_dataset = mnist.MNIST(root='./mnist_test', train=False, transform=ToTensor(), download=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = torch.load('./lenet_ckpt/mnist_0.98.pt')

    feature_wise_attack = PGD(model = model)


    all_correct_num = 0
    all_sample_num = 0
    model.eval()

    
    acc_array_epoch = []
    correct_array_epoch = []
    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.cuda()
        adv_images_array, clean_feature_map, adv_feature_map_array = feature_wise_attack.forward(test_x, test_label.cuda())
        # print(torch.tensor( [item.cpu().detach().numpy() for item in adv_feature_map_array] ).shape)
        acc_array_channel = []
        correct_num_array = []
        for adv_images in adv_images_array:
            predict_y, _ = model(adv_images)
            predict_y = np.argmax(predict_y.cpu().detach(), axis=-1)
            # print(predict_y.shape)
            current_correct_num = predict_y == test_label
            all_correct_num = np.sum(current_correct_num.numpy(), axis=-1)
            correct_num_array.append(all_correct_num)
            all_sample_num = current_correct_num.shape[0]
            acc = all_correct_num / all_sample_num
            acc_array_channel.append(acc)

        sorted_feature_wise_acc_idx = np.argsort(np.array(acc_array_channel))
        for idx, sorted_idx in enumerate(sorted_feature_wise_acc_idx):
            idx *= 2
            plt.subplot(16, 2, idx+1)
            plt.imshow(np.array(clean_feature_map[0, sorted_idx].cpu().detach()))
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.subplot(16, 2, idx+2)
            plt.imshow(np.array(adv_feature_map_array[sorted_idx][0].cpu().detach()))
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        plt.savefig('./test.png')

        break


        acc_array_epoch.append(acc_array_channel)
        correct_array_epoch.append(correct_num_array)


    # print(np.array(acc_array_epoch))
    # print(np.array(acc_array_epoch).shape)
    sorted_feature_wise_acc = np.argsort(np.mean(np.array(acc_array_epoch), axis=0))
    print(sorted_feature_wise_acc)
    print(np.mean(np.array(acc_array_epoch), axis=0))
    print(np.sum(np.array(correct_array_epoch), axis=0))

