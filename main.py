import argparse
import datetime
import numpy as np
import time
import torch
import GPUtil
from tqdm import tqdm
import mlflow
from contextlib import suppress
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch import optim
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

import timm
from timm.utils import NativeScaler, get_state_dict

from engine import train_one_epoch, evaluate
import utils
from dataset import get_data

from FunctionalMamba.functional_mamba.model_hub import *


def get_args_parser():
    parser = argparse.ArgumentParser('Model Training')
    parser.add_argument('--epoch', type=int, default=200, help='Epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--model_name', type=str, default='FunctionalMamba', help='Model name')
    parser.add_argument('--dataset_name', type=str, default='HCP_dfnc')
    parser.add_argument('--image_format', type=str, default='2DT')
    parser.add_argument('--target_name', type=str, required=False, default='sex')
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--num_classes', type=int, default=1, required=False, help='num_classes')
    parser.add_argument('--loss', type=str, default='bce_with_logit')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--fold_ids', type=int, nargs='+', help='train the given folds')
    parser.add_argument('--cuda_id', type=int, help='cuda device id', default=0)

    parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint or pretrain')
    parser.add_argument('--resume_path', type=str, required=False, help='saved checkpoint path')
    parser.add_argument('--add_class_weight', action='store_true', default=False, help='criterion class weight')
    parser.add_argument('--clip_grad', action='store_true', default=True, help='clip grad')
    parser.add_argument('--save_model', action='store_true', default=False, help='save checkpoint')
    parser.add_argument('--enable_mlflow', action='store_true', default=False)
    
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--if_amp', action='store_true', default=False)
    return parser


def main(args):
    args.model_name += '_' + args.dataset_name + '-' + args.target_name
    print(args)

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = torch.device(args.cuda_id)

    seed = args.seed + utils.get_rank()
    if args.distributed:
        utils.init_distributed_mode(args)
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        print(f"Start running basic DDP example on rank {rank}.")
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    if args.enable_mlflow and utils.get_rank() == 0:
        # mlflow.login()
        mlflow.set_tracking_uri('')
        mlflow.set_experiment('/'+ args.dataset_name + '_' + args.target_name)
        mlflow.set_tag('mlflow.runName', args.model_name)
        mlflow.set_tag('model_name', args.model_name)
        # mlflow.autolog(extra_tags={'model_name': args.model_name, 'dataset': args.dataset_name, 'target': args.target_name})
        mlflow.log_params(vars(args))
        utils.log_all_files(args)

    stats = {}
    target_metric = 'auc' if args.task == 'classification' else 'mae'
    args.fold_ids = [0, 1, 2, 3, 4] if args.fold_ids == None else args.fold_ids
    for fold_id in args.fold_ids:
        data_loader_train, data_loader_test = get_data(fold_id, args, data_name=args.dataset_name, is_ddp=args.distributed, is_test_ddp=args.distributed, target=args.target_name, format=args.image_format)

        model = mambaf_multi_st_base_v1_004(in_chans=1, num_classes=args.num_classes, bimamba_type=['bi_st']).cuda()
        # with open(f'checkpoints/trained_models/{args.model_name}_model_summary.txt', 'w') as f:
        #     f.write(str(summary(model)))
        if args.enable_mlflow and utils.get_rank() == 0:
            mlflow.log_artifact(f'checkpoints/trained_models/{args.model_name}_model_summary.txt')
        
        if args.resume:
            root = f'checkpoints/contrast_pretrained/EHBS_dfnc/{args.resume_path}/200ep_{args.resume_path}.pth'
            state_dict = torch.load(root, map_location=device)['model_state_dict']
            state_dict = OrderedDict({k: v for k, v in state_dict.items() if 'head' not in k})
            model.load_state_dict(state_dict, strict=False)

        model = model.to(device)

        model_without_ddp = model
        if args.distributed:
            model = DDP(model, device_ids=[device_id])
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

        amp_autocast = suppress
        loss_scaler = "none"
        if args.if_amp:
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()

        if args.task == 'classification':
            if args.num_classes == 1:
                if args.add_class_weight:
                    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
                    print('use class weight')
                else:
                    criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            criterion = nn.MSELoss()

        if args.task == 'classification':
            best_stats = {"acc": (0, ), 'f1': (0, ), 'auc': (0, ), 'ap': (0, ), 'recall': (0, )}
        else:
            best_stats = {'mae': (1e6, ), 'r2': (0, ), 'r': (0, )}
        accumulated_metrics = {}
        
        epochs = tqdm(range(args.epoch))
        for e in epochs:
            
            if args.distributed:
                data_loader_train.batch_sampler.sampler.set_epoch(e)
            train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, e, loss_scaler, amp_autocast, args.clip_grad, args=args)

            scheduler.step()

            if train_stats == None:
                break

            epochs.set_description(f"{args.model_name} fold:{fold_id} rank:{utils.get_rank()} - average loss: {round(train_stats['loss'], 3)}")

            test_stats, out_pred = evaluate(data_loader_test, model, device, amp_autocast, args, criterion)

            if utils.get_rank() == 0:
                best_stats = utils.save_best(test_stats, best_stats, e, fold_id, out_pred, model, args)

            all_train_stats = [{} for _ in range(utils.get_world_size())]
            if dist.is_initialized():
                dist.all_gather_object(all_train_stats, train_stats)
                train_stats = {k: sum(d[k] for d in all_train_stats) / len(all_train_stats) for k in train_stats.keys()}

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': e,
                     'n_parameters': n_parameters}
            
            if args.enable_mlflow:
                utils.accumulate_metrics(log_stats, fold_id, accumulated_metrics)
        
        print(f"best performance for fold {fold_id}:", best_stats)
        
        best_stats_to_save = {'best_' + k + f'_{fold_id}': round(v, 3) for k, v in best_stats[target_metric][1].items()}
        best_stats = {k: round(v, 3) for k, v in best_stats[target_metric][1].items()}  # ACC as the primary metric
        if utils.get_rank() == 0:
            if args.enable_mlflow:
                utils.log_accumulated_metrics(args, accumulated_metrics)
                mlflow.log_metrics(best_stats_to_save, step=fold_id)
            for k in best_stats:
                if k in stats:
                    stats[k] += best_stats[k]
                else:
                    stats[k] = best_stats[k]
    
    if args.distributed:
        dist.destroy_process_group()

    for k in stats: 
        print(f"average {k}: ", stats[k] / len(args.fold_ids))
        stats[k] = round(stats[k] * 100, 5) if args.task == 'classification' else round(stats[k], 5)

    with open(f'checkpoints/trained_predictions/{args.dataset_name}/{args.model_name}/job_status.txt', 'a') as f:
        f.write(f'finished folds: {args.fold_ids}\n')

    mlflow.log_metrics({'ave_' + k: v / len(args.fold_ids) for k, v in stats.items()}, step=0)


if __name__ =='__main__':
    args = get_args_parser().parse_args()
    main(args)

