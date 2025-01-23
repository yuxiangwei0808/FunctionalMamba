import io
import os
import time
from collections import defaultdict, deque
import datetime
import random
import monai.transforms
import numpy as np
import mlflow
import json
import monai
from sklearn.metrics import precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

import torch
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / (self.count + 1)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('{} Total time: {} ({:.4f} s / it)'.format(
        #     header, total_time_str, total_time / len(iterable)))



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def save_model(model, epoch, test_stats, path, model_name, optimizer=None):
    print(f"saving: {test_stats}")
    if not os.path.isdir(path):
        os.makedirs(path)
    save_files = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    save_files.update(test_stats) if test_stats else ...
    torch.save(save_files, os.path.join(path, model_name))


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k

    X_train, y_train = None, None
    for j in range(k):
        if j == k - 1:
            idx = slice(j * fold_size, X.shape[0])
        else:
            idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)
            y_train = np.concatenate((y_train, y_part), axis=0)
    return X_train, y_train, X_valid, y_valid


def vec2mat(vec, full=True):
    # vec (B, C, N), mat (B, C, H, W)
    # returns the correlation matrix from a vector
    # if full is True, returns the symmetric matrix
    N = vec.shape[-1]
    n = int(1/2 + np.sqrt(1 + 8 * N) / 2)
    mat = np.full((vec.shape[0], vec.shape[1], n, n), np.nan)
    
    i = 0
    for c in range(0, n):
        for r in range(c+1, n):
            mat[:, :, r, c] = vec[:, :, i]
            i += 1

    if full:
        mat = np.flip(np.rot90(mat, axes=(-2, -1)), axis=-2)
        i = 0
        for c in range(0, n):
            for r in range(c+1, n):
                mat[:, :, r, c] = vec[:, :, i]
                i += 1
    mat[np.isnan(mat)] = 0
    return mat.astype(np.float32)


def vec2img(data, sum_time=False, target_shape=(64, 64), format='2D'):
    print("transforming images...")
    data = vec2mat(data)
    data = torch.tensor(data)
    data = torch.nn.Upsample(target_shape[:2], mode='nearest')(data)
    if sum_time and len(target_shape) > 2:
        # data = np.sum(data, axis=1, keepdims=True)
        data = F.interpolate(data.permute(0, 2, 3, 1), scale_factor=(1, target_shape[0] / data.shape[1]))
    if format == 'T2D':
        data = data.unsuqeeze(1)
    print("transformed shape: ", data.shape)
    return np.array(data)


def select_subject(x, target=0, num=90):
    # select num target subjects from x and return indices
    ind_target = set(np.where(x==target)[0])
    ind_other = set(np.arange(0, len(x), 1)) - ind_target
    return list(ind_other) + sorted(random.sample(ind_target, num))


def compare_performance(root, mode=1):
    if mode:
        for i in range(5):
            a = torch.load(f'checkpoints/trained_models/{root}/{root}_{i}_a.pth')
            f = torch.load(f'checkpoints/trained_models/{root}/{root}_{i}_f.pth')
            print(a['accuracy'], a['f1'])
            print(f['accuracy'], f['f1'])
            print('=================')
    else:
        acc = 0
        f1 = 0
        for i in range(5):
            x = torch.load(f'checkpoints/trained_models/{root}/{root}_{i}_f.pth')
            acc += x['accuracy']
            f1 += x['f1']
        print(acc / 5, f1 / 5)
    

def save_best(test_stats, best_stats, e, fold_id, target_predraw_pred, model, args, base_metric=None):
    checkpoint_root = f'checkpoints/trained_models/{args.dataset_name}/{args.model_name}'
    preds_root = f'checkpoints/trained_predictions/{args.dataset_name}/{args.model_name}'
    if not os.path.isdir(checkpoint_root):
        os.makedirs(checkpoint_root)
    if not os.path.isdir(preds_root):
        os.makedirs(preds_root)

    for k in test_stats:
        if (k == 'mae' and test_stats[k] < best_stats[k][0]) or (k != 'mae' and test_stats[k] > best_stats[k][0]):
            best_stats[k] = (test_stats[k], test_stats)
            print(f"best {k} on epoch {e}: ", best_stats[k])
            if args.save_model:
                save_model(model, e, test_stats, checkpoint_root, f'{args.model_name}_{fold_id}_{k}.pth')
            saving_dict = {'true': target_predraw_pred[0], 'pred': target_predraw_pred[1]}
            saving_dict.update(**test_stats)
            np.save(os.path.join(preds_root, f'{args.model_name}_{fold_id}_{k}.npy'), saving_dict)
            if args.enable_mlflow:
                log_misclassified_samples(target_predraw_pred[0], target_predraw_pred[2], e, fold_id, args)            

    return best_stats


def z_score_normalize(input, dim):
    print('data z-score normalized!')
    std, mean = torch.std_mean(input.float(), dim=dim, keepdim=True)
    return (input - mean) / (std + 1e-6)


def accumulate_metrics(log_stats, fold_id, accumulated_metrics):
    epoch = log_stats['epoch']
    for key, value in log_stats.items():
        metric_key = f"{key}_{fold_id}"
        if metric_key not in accumulated_metrics:
            accumulated_metrics[metric_key] = []
        # Store metric value along with its epoch
        accumulated_metrics[metric_key].append((epoch, value))


def log_accumulated_metrics(args, accumulated_metrics):
    if get_rank() == 0:
        for metric_key, values in accumulated_metrics.items():
            for epoch, value in values:
                mlflow.log_metric(metric_key, value, epoch)


def log_misclassified_samples(targets, predictions, epoch, fold, args):
    misclassified_indices = [i for i, (pred, true) in enumerate(zip(predictions.squeeze(), targets.squeeze())) if pred != true]
    misclassified_info = {
        "epoch": epoch,
        "fold": fold,
        "misclassified_indices": misclassified_indices,
        "targets": targets.squeeze().tolist(),
        "predictions": predictions.squeeze().to(torch.int).tolist(),
    }
    if not os.path.isdir(f'checkpoints/artifacts/{args.model_name}'):
        os.mkdir(f'checkpoints/artifacts/{args.model_name}')
    file_name = f'checkpoints/artifacts/{args.model_name}/{args.model_name}_{fold}.json'
    with open(file_name, "w") as f:
        json.dump(misclassified_info, f)
    mlflow.log_artifact(file_name)


def log_all_files(args):
    if not os.path.isdir(f'checkpoints/artifacts/{args.model_name}'):
        os.makedirs(f'checkpoints/artifacts/{args.model_name}')
    mlflow.log_artifact('main.py', 'scripts')
    mlflow.log_artifact('dataset.py', 'scripts')
    mlflow.log_artifact('engine.py', 'scripts')
    mlflow.log_artifact('utils.py', 'scripts')
    mlflow.log_artifact('mambaf/mambaf/', 'scripts')


### SAM ###
def disable_running_stats(model):
    # disable bn (used for SAM)
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
######

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def dim_reducer(data, reducer='lle'):
    new_data = []
    if reducer == 'pca':
        r = PCA(n_components=30)
    elif reducer == 'lle':
        r = LocallyLinearEmbedding(n_components=30, n_neighbors=25)
    
    for x in data:
        x = x.reshape(x.shape[0], -1).permute(1, 0)
        x = r.fit_transform(x)
        new_data.append(x.transpose().reshape(x.shape[-1], 64, 64)[None, :])
    new_data = np.concatenate(new_data, 0)
    return torch.tensor(new_data, dtype=torch.float32)
    

def fmri_transform(img, target_shape=(64, 64, 64), norm=True):
    img = torch.tensor(img.get_fdata())
    img = F.interpolate(img, scale_factor=(1, 20 / img.shape[-1]))  # downsample the temporal dimension
    img = img.permute(3, 0, 1, 2)
    img = img.unsqueeze(0)  # add a channel dimension, and needed for nn.Upsample
    # img = monai.transforms.ResizeWithPadOrCrop(target_shape, mode='reflect')(img)
    img = torch.nn.Upsample(target_shape, mode='trilinear', align_corners=True)(img)
    if norm:
        img = monai.transforms.NormalizeIntensity()(img)
    img = img.permute(0, 2, 3, 4, 1)  # B H W L T
    return img


class Aug:
    def __init__(self, mix_type):
        self.mix_type = mix_type
        
    def perform_aug(self, matrix):
        B, C, H, W, T = matrix.shape
        if self.mix_type == -1:
            return matrix

        rand_noise = monai.transforms.RandGaussianNoise(prob=0.25, std=0.5)
        rand_smooth = monai.transforms.RandGaussianSmooth(sigma_x=(0, 0.5), sigma_y=(0, 0.5), prob=0.1)
        comp = monai.transforms.Compose([rand_noise, rand_smooth])

        for b in range(B):
            seed = torch.randint(0, 1000000, (1,)).item()
            for t in range(T):
                comp.set_random_state(seed)
                matrix[b, ..., t] = self.mix(matrix[b, ..., t], seed)
                matrix[b, ..., t] = comp(matrix[b, ..., t])
        return matrix
    
    def _gen_rand_box(self, seed, minmax=(0.1, 0.6), num_comps=None, L=64):
        assert len(minmax) == 2
        np.random.seed(seed)
        num_comps = np.random.randint(int(L * minmax[0]), int(L * minmax[1])) if not num_comps else num_comps
        comps = np.random.randint(0, L, size=num_comps)
        return comps
    
    def mix(self, matrix, seed):
        # matrix: C H W
        comps = self._gen_rand_box(seed)
        target_comps = self._gen_rand_box(seed, num_comps=len(comps))
        if 0 == self.mix_type:  # cover
            matrix[:, target_comps, :] = matrix[:, comps, :]
        elif 1 == self.mix_type:  # add
            matrix[:, target_comps, :] += matrix[:, comps, :]
        elif 2 == self.mix_type:  # erase
            matrix[:, target_comps, :] = 1e-6
        elif 3 == self.mix_type:  # exchange
            tmp = matrix[:, target_comps, :]
            matrix[:, target_comps, :] = matrix[:, comps, :]
            matrix[:, target_comps, :] = tmp
        # 4: do nothing
        return matrix
    
    def process_dataloader(self, data_loader, temporal_split_type):
        processed_x1, processed_x2 = [], []
        for samples, _ in data_loader:
            T = samples.shape[-1]
            if temporal_split_type == 0:  # no split
                x1, x2 = samples.clone(), samples
            elif temporal_split_type == 1:  # [x1 x2]
                x1 = samples[..., :T // 2]
                x2 = samples[..., T // 2:]
            elif temporal_split_type == 2:  # alternating
                odd_index, even_index = torch.arange(1, T, 2), torch.arange(0, T, 2)
                x1, x2 = torch.index_select(samples, -1, even_index), torch.index_select(samples, -1, odd_index)
            elif temporal_split_type == 3: # local - global
                x1, x2 = samples[..., :T // 2], torch.index_select(samples, -1, torch.arange(0, T, 2))
            else:
                raise NotImplementedError                
            
            x1, x2 = self.perform_aug(x1), self.perform_aug(x2)
            processed_x1.append(x1)
            processed_x2.append(x2)
            
        return processed_x1, processed_x2