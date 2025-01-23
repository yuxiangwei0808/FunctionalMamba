import argparse
import datetime
import numpy as np
import time
import torch
import GPUtil
from tqdm import tqdm
import mlflow
from contextlib import suppress
import os

import torch
import torch.backends.cudnn as cudnn
from torch import optim
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

import timm
from timm.utils import NativeScaler, get_state_dict

from engine import pretrain_contrastive
import utils
from dataset import get_data
from lr_scheduler import CosineAnnealingWarmUpRestarts

from FunctionalMamba.functional_mamba.model_hub import *


def get_args_parser():
    parser = argparse.ArgumentParser('Model Training')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--model_name', type=str, default='BNT', help='Model name')
    parser.add_argument('--dataset_name', type=str, default='EHBS_dfnc')
    parser.add_argument('--image_format', type=str, default='2D')
    parser.add_argument('--target_name', type=str, required=False, default='AsymAD')
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--num_classes', type=int, default=1, required=False, help='num_classes')
    parser.add_argument('--loss', type=str, default='bce_with_logit')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--cuda_id', type=int, help='cuda device id', default=0)

    parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint or pretrain')
    parser.add_argument('--resume_path', type=str, required=False, help='saved checkpoint path')
    parser.add_argument('--clip_grad', action='store_true', default=True, help='clip grad')
    parser.add_argument('--save_model', action='store_true', default=True, help='save checkpoint')
    parser.add_argument('--enable_mlflow', action='store_true', default=False)
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--if_amp', action='store_true', default=False)

    parser.add_argument('--aug_type', type=int, choices=[0, 1, 2, 3, 4], default=-1)
    parser.add_argument('--temporal_split_type', type=int, choices=[0, 1, 2], default=-1)
    return parser


def main(args):
    args.model_name += '_' + args.dataset_name + '-' + args.target_name
    print(args)

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = torch.device(args.cuda_id)

    seed = args.seed + utils.get_rank()
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    if args.enable_mlflow and utils.get_rank() == 0:
        mlflow.login()
        mlflow.set_experiment('/pretrain-'+ args.dataset_name + '_' + args.target_name)
        mlflow.set_tag('mlflow.runName', args.model_name)
        mlflow.set_tag('model_name', args.model_name)
        # mlflow.autolog(extra_tags={'model_name': args.model_name, 'dataset': args.dataset_name, 'target': args.target_name})
        mlflow.log_params(vars(args))
        utils.log_all_files(args)

    stats = {}
    data_loader = get_data(0, args, fold=-1, data_name=args.dataset_name, is_ddp=args.distributed, is_test_ddp=args.distributed, target=args.target_name, format=args.image_format)

    model = mambaf_multi_st_base_v1_004(in_chans=1, num_classes=args.num_classes, bimamba_type=['bi_st'], pretrain=True).cuda()

    with open(f'checkpoints/trained_models/{args.model_name}_model_summary.txt', 'w') as f:
        f.write(str(summary(model)))
    if args.enable_mlflow:
        mlflow.log_artifact(f'checkpoints/trained_models/{args.model_name}_model_summary.txt')
    os.remove(f'checkpoints/trained_models/{args.model_name}_model_summary.txt')

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    # amp about
    amp_autocast = suppress
    loss_scaler = "none"
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    if args.task == 'classification':
        if args.num_classes == 1: 
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.MSELoss()
    
    pretrain_contrastive(model, optimizer, scheduler, data_loader, device, temporal_split_type=args.temporal_split_type, args=args)

if __name__ =='__main__':
    args = get_args_parser().parse_args()
    main(args)