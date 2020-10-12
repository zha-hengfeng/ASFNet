
import random
import numpy as np
import torch
import torch.nn as nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


import matplotlib.pyplot as plt
# visual
def draw_loss(start_epoch, epoch, lossTr_list, savedir):
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(start_epoch, epoch + 1), lossTr_list)
    ax1.set_title("Average training loss vs epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Current loss")
    plt.savefig(savedir + "loss_vs_epochs.png")
    plt.clf()
    plt.close('all')

# visual
def draw_miou(epoches, mIOU_val_list, savedir):
    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(epoches, mIOU_val_list, label="Val IoU")
    ax2.set_title("Average IoU vs epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Current IoU")
    plt.legend(loc='lower right')
    plt.savefig(savedir + "iou_vs_epochs.png")
    plt.clf()
    plt.close('all')