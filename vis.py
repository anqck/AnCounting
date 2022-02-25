import datetime
import random
import time
import argparse
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('agg')

from dataset import build_dataset

MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training AnNet', add_help=False)


    # dataset parameters
    parser.add_argument('--dataset_file', default='FSC147')
    parser.add_argument('--data_path', default='../Data/',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

def visualize_output_and_save(input_, output, boxes, save_path, figsize=(20, 12), dots=None):
    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """

    # get the total count
    pred_cnt = output.sum().item()
    boxes = boxes.squeeze(0)

    boxes2 = []
    for i in range(0, boxes.shape[0]):
        y1, x1, y2, x2 = int(boxes[i, 1].item()), int(boxes[i, 2].item()), int(boxes[i, 3].item()), int(
            boxes[i, 4].item())
        roi_cnt = output[0,0,y1:y2, x1:x2].sum().item()
        boxes2.append([y1, x1, y2, x2, roi_cnt])

    img1 = format_for_plotting(denormalize(input_))
    output = format_for_plotting(output)

    fig = plt.figure(figsize=figsize)

    # display the input image
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(img1)

    for bbox in boxes2:
        y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)

    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
        # ax.scatter(dots[:,0], dots[:,1], c='black', marker='+')
        ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        ax.set_title("Input image")

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))

    img2 = 0.2989*img1[:,:,0] + 0.5870*img1[:,:,1] + 0.1140*img1[:,:,2]
    ax.imshow(img2, cmap='gray')
    ax.imshow(output, cmap=plt.cm.viridis, alpha=0.5)


    # display the density map
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output)
    # plt.colorbar()

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ret_fig = ax.imshow(output)
    for bbox in boxes2:
        y1, x1, y2, x2, roi_cnt = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.text(x1, y1, '{:.2f}'.format(roi_cnt), backgroundcolor='y')

    fig.colorbar(ret_fig, ax=ax)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close()

def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    Args:
        tensor (torch.Tensor, torch.float32): Image tensor
    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively
    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()

def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    """Reverses the normalisation on a tensor.
    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.
    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean
    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor
    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)
    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)

    return denormalized

def main(args):

    loading_data, collate_fn_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set, test_set = loading_data(args.data_path)

    # create the sampler used during training
    # sampler_train = torch.utils.data.SequentialSampler(train_set)
    # data_loader_train = DataLoader(train_set, 1, drop_last=False, num_workers=0)

    print(train_set[0])
    samples, boxes,gt_density,points = train_set[2]

    visualize_output_and_save(samples,gt_density.unsqueeze(0), boxes, 'a.png' )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)