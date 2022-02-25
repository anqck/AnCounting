from typing import Iterable
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from utils import extract_features

def train_one_epoch(regressor: torch.nn.Module, backbone: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # metric_logger = utils.MetricLogger(delimiter="  ")
    train_mae = 0
    train_rmse = 0
    train_loss = 0

    # print(len(data_iterator))
    pbar = tqdm(data_loader)
    for samples, boxes,gt_density,points in pbar: 
        samples = samples.to(device)
        boxes = boxes.to(device)
        gt_density = gt_density.to(device)
        
        # print(samples.shape,boxes.shape)

        # boxes = [{k: v.to(device) for k, v in t.items()} for t in boxes]
        # print(boxes)

        with torch.no_grad():
            features = extract_features(backbone, samples, boxes)

        # print(features[0][0][0])
        # print(features[1][0][0])
        features.requires_grad = True
        optimizer.zero_grad()
        output = regressor(features)

        # print(gt_density.sum(), output.sum())
        # print(output.shape, gt_density.shape)
        #if image size isn't divisible by 8, gt size is slightly different from output size
        if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(output.shape[2],output.shape[3]),mode='bilinear')
            new_count = gt_density.sum().detach().item()
            if new_count > 0: gt_density = gt_density * (orig_count / new_count)

        loss = criterion(output, gt_density)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_cnt = torch.sum(output, axis = (1,2,3)).detach()
        gt_cnt = torch.sum(gt_density, axis = (1,2,3)).detach()
        cnt_err = abs(pred_cnt - gt_cnt)
        # print(pred_cnt)
        train_mae += cnt_err.sum()
        # print(train_mae)
        train_rmse += (cnt_err ** 2).sum()

        pbar.set_description('actual-predicted: {:6.1f}, {:6.1f}, error: {:6.1f}'.format( gt_cnt[0], pred_cnt[0], abs(pred_cnt[0] - gt_cnt[0])))
        # print('actual-predicted: {:6.1f}, {:6.1f}, error: {:6.1f}'.format( gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt)))

    train_loss = train_loss / len(data_loader.dataset)
    train_mae = (train_mae / len(data_loader.dataset))
    train_rmse = (train_rmse / len(data_loader.dataset))**0.5
    return train_loss,train_mae,train_rmse

def eval(args, regressor: torch.nn.Module, backbone: torch.nn.Module, data_loader: Iterable, device: torch.device):
    cnt = 0
    SAE = 0 # sum of absolute errors
    SSE = 0 # sum of square errors

    print("Evaluation on {} data".format(args.test_split))
    # im_ids = data_split[args.test_split]

    pbar = tqdm(data_loader)
    for samples, boxes,gt_density,points in pbar:
        samples = samples.to(device)
        boxes = boxes.to(device)

        # print(samples.shape, boxes.shape)
        with torch.no_grad():
            output = regressor(extract_features(backbone, samples, boxes))


        # print("AAAAAA",output.sum())
        # assert 1==0
        # for i in range(args.batch_size):
            # print(boxes.shape)
            # print(points)
        gt_cnt = len(points)
        pred_cnt = torch.sum(output).item()
        cnt = cnt + 1
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2

        pbar.set_description(' actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.format( gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
        # print("")

    print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
    return SAE/cnt, (SSE/cnt)**0.5