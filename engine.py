"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import numpy as np
from tqdm import tqdm
import mlflow

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from timm.layers.classifier import ClassifierHead

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, r2_score, average_precision_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import utils
from loss import *


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 1.0,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.num_classes > 1:
            targets = targets.squeeze().long()
    
        with amp_autocast():
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss_value = loss.mean().item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            return

        # this attribute is added by timm on one optimizer (adahessian)
        if loss_scaler != 'none':
            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            optimizer.zero_grad()
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}   


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, args, criterion):
    model.eval()

    targets = []
    preds_raw = []

    for images, target in data_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        output, target = output.cpu(), target.cpu()
        
        targets.append(target)
        preds_raw.append(output)
    
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
             
    # gather the stats from all processes
    targets = torch.tensor(np.concatenate(targets)).to(device)
    preds_raw = torch.tensor(np.concatenate(preds_raw)).to(device)

    if utils.is_dist_avail_and_initialized():
        if dist.get_rank() == 0:
            gathered_targets= [torch.zeros_like(targets) for _ in range(dist.get_world_size())]
            gathered_preds_raw= [torch.zeros_like(preds_raw) for _ in range(dist.get_world_size())]
            dist.gather(targets, gather_list=gathered_targets, dst=0)
            dist.gather(preds_raw, gather_list=gathered_preds_raw, dst=0)
            targets, preds_raw = torch.cat(gathered_targets, 0), torch.cat(gathered_preds_raw, 0)
        else:
            dist.gather(targets)
            dist.gather(preds_raw)

    pred = torch.clone(preds_raw).cpu()
    targets, preds_raw = np.array(targets.cpu()), np.array(preds_raw.cpu())

    if args.task == 'classification':
        if args.num_classes == 1 and pred.shape[-1] == 1:
            pred = torch.sigmoid(pred)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
        else:
            pred = torch.softmax(pred, dim=-1)
            pred = torch.argmax(pred, dim=-1)

    if args.task == 'classification':
        mode = 'binary' if args.num_classes <= 2 else 'micro'
        multi_class = 'raise' if args.num_classes <= 2 else 'ovo'
        preds_roc = np.array(torch.sigmoid(torch.tensor(preds_raw))) if args.num_classes == 1 else np.array(torch.softmax(torch.tensor(preds_raw), -1))
        targets_roc = np.array(F.one_hot(torch.tensor(targets.squeeze(), dtype=int))) if args.num_classes == 2 else targets
        test_stats = {"acc": accuracy_score(targets, pred), "f1": f1_score(targets, pred, average=mode),
                       "auc": roc_auc_score(targets_roc, preds_roc, multi_class=multi_class), 'ap': average_precision_score(targets_roc, preds_roc)}
    else:
        targets, pred = targets.squeeze(), pred.squeeze()
        test_stats = {'mae': mean_absolute_error(targets, pred), 'r2': r2_score(targets, pred), 'r': pearsonr(targets, np.array(pred))[0]}

    return test_stats, [targets, preds_raw, pred]


def pretrain_contrastive(model, optimizer, scheduler, data_loader, device, temporal_split_type, args, max_norm=1.0):
    model.train()
    pretrain_loss = []
    loss_scaler = GradScaler()
    criterion = NTXentLoss(device, args.batch_size, temperature=0.1, use_cosine_similarity=False)
    # output_heads = [ClassifierHead(in_features=feat, num_classes=1).to(device) for feat in [192]]
    # output_heads = [(nn.Linear(feat, 1).to(device), nn.Linear(feat, 1).to(device)) for feat in [192]]
    output_heads = [lambda x: x for _ in range(1)]

    aug = utils.Aug(mix_type=args.aug_type)
    epochs = tqdm(range(args.epoch))

    processed_samples  = aug.process_dataloader(data_loader, temporal_split_type)
    # processed_samples_lg = aug.process_dataloader(data_loader, 3)
    

    for e in epochs:
        epoch_loss = 0.
        for i in range(len(processed_samples[0])):
            loss = 0.
            
            x1, x2 = processed_samples[0][i].to(device), processed_samples[1][i].to(device)                
            # instance
            out1 = model(x1)
            out2 = model(x2)
            for stage_id, (a, b) in enumerate(zip(out1, out2)):
                loss += criterion(output_heads[stage_id](a), output_heads[stage_id](b))


            epoch_loss += loss.item()
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

        epochs.set_description(f"pretraining {args.model_name} rank:{utils.get_rank()} - loss: {epoch_loss:.2f}")
        pretrain_loss.append(epoch_loss)
        scheduler.step()

    ckp_path = f'checkpoints/contrast_pretrained/{args.dataset_name}/{args.model_name}'
    utils.save_model(model, e, None, ckp_path, f'{args.epoch}ep_' + args.model_name + '.pth')

    plt.plot()
    plt.plot(range(args.epoch), pretrain_loss)
    plt.savefig('loss_curve.png')