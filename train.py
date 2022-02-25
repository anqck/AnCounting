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
import torch.nn.functional as F
from dataset import build_dataset
from model import Resnet50FPN, CountRegressor
from utils import weights_normal_init, set_random_seed
from engine import train_one_epoch, eval

import wandb


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training AnNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument("-ts", "--test-split", type=str, default='val', choices=["train", "test", "val"], help="what data split to evaluate on on")

    # dataset parameters
    parser.add_argument('--dataset_file', default='FSC147')
    parser.add_argument('--data_path', default='../Data/',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    parser.add_argument('--wandb', action='store_true', help='wandb')
    

    return parser

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)

    # fix the seed for reproducibility
    set_random_seed(args.seed)
    
    
    if args.wandb:
        wandb.init(project="famnet", entity="anqck", tags=["AnCounting"])
        wandb.define_metric("Val MAE", summary="min")
        wandb.define_metric("Val RMSE", summary="min")
        run_name = wandb.run.name
        

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    device = torch.device('cuda')

    # create the dataset
    loading_data, collate_fn_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set, test_set = loading_data(args.data_path)

    # create the sampler used during training
    sampler_train = torch.utils.data.SequentialSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)
    sampler_test = torch.utils.data.SequentialSampler(test_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=False)
    # the dataloader for training
    # data_loader_train = DataLoader(train_set, args.batch_size, drop_last=False, num_workers=args.num_workers)

    data_loader_train = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    # data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=args.num_workers,collate_fn=collate_fn_data)


    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val, \
                                drop_last=False, num_workers=args.num_workers)

    criterion = nn.MSELoss().to(device)
    resnet50_conv = Resnet50FPN().to(device)
    resnet50_conv.eval()

    regressor = CountRegressor(6, pool='mean').to(device)
    weights_normal_init(regressor, dev=0.001)
    regressor.train()
    optimizer = optim.Adam(regressor.parameters(), lr = args.lr)

    best_mae, best_rmse = 1e7, 1e7
    stats = list()
    for epoch in range(0, args.epochs):

        # t1 = time.time()

        regressor.train()
        train_loss,train_mae,train_rmse = train_one_epoch(regressor,resnet50_conv, criterion, data_loader_train, optimizer, device, epoch)

        regressor.eval()
        val_mae,val_rmse = eval(args, regressor,resnet50_conv,data_loader_val, device)

        stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
        if best_mae >= val_mae:
            best_mae = val_mae
            best_rmse = val_rmse
            model_name = args.output_dir + '/' + run_name + ".pth"
            torch.save(regressor.state_dict(), model_name)

        # # Log loss
        if args.wandb:
            wandb.log({"Epoch": epoch+1,
                        "Avg. Epoch Loss:": stats[-1][0],
                        "Train MAE": stats[-1][1],
                        "Train RMSE": stats[-1][2],
                        "Val MAE": stats[-1][3],
                        "Val RMSE": stats[-1][4]
                        })

        print("Epoch {}, Avg. Epoch Loss: {} Train MAE: {} Train RMSE: {} Val MAE: {} Val RMSE: {} Best Val MAE: {} Best Val RMSE: {} ".format(
              epoch+1,  stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse))

        # total time for training
        # total_time = time.time() - t1
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
