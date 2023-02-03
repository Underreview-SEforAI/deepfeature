import torch
import numpy as np
import matplotlib.pyplot as plt


def feature_map_plot(vulnerable_feature_map_idx, clean_feature_map, adv_feature_map):

    # sorted_feature_wise_acc_idx = np.argsort(np.array(acc_array_channel))
    # clean_feature_map = clean_feature_map[vulnerable_feature_map_idx]
    # adv_feature_map_array = adv_feature_map[vulnerable_feature_map_idx]
    for idx, sorted_idx in enumerate(vulnerable_feature_map_idx):
        idx *= 2
        plt.subplot(5, 2, idx+1)
        plt.imshow(np.array(clean_feature_map[0][sorted_idx].cpu().detach()))
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.subplot(5, 2, idx+2)
        plt.imshow(np.array(adv_feature_map[0][sorted_idx].cpu().detach()))
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    plt.savefig('./feature_map_visualize.png')