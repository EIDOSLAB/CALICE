import torch.nn as nn
import torch.optim as optim

import torch
import shutil

from .engine import test_epoch,train_one_epoch, compress_one_epoch, AverageMeter, lambda_interpolated
from .engine_rdp import beta_interpolated
from .loss import RateDistortionLoss, RateDistortionLoss_withLPIPS, RateDistortionPerceptionLoss
from .dataset import TestKodakDataset
from .functions import compute_metrics, compute_msssim, compute_psnr, compute_lpips
from .plot import plot_rate_distorsion, plot_rate_distortion_perception
from .dataset import SquarePad
from custom_comp.datasets import ImageFolder

import random
import os
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np

import torch.nn.functional as F
import numpy as np



def configure_optimizers(net, args, assert_intersection = True, opt = 'adam'):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    if assert_intersection:
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0


    params = [params_dict[n] for n in sorted(parameters)]
    if len(params) > 0:
        print('Creating Optimizer for main network')
        if opt == 'adam':
            print('Adam')
            optimizer = optim.Adam(
                (params_dict[n] for n in sorted(parameters)),
                lr=args.learning_rate,
            )
        else:
            print('SGD')
            optimizer = optim.SGD(
                (params_dict[n] for n in sorted(parameters)),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=0.0
            )
    else:
        optimizer = None

    aux_params = [params_dict[n] for n in sorted(aux_parameters)]
    if len(aux_params) > 0:
        print('Creating Optimizer for aux network')
        if opt == 'adam':
            print('Adam')
            aux_optimizer = optim.Adam(
                (params_dict[n] for n in sorted(aux_parameters)),
                lr=args.aux_learning_rate,
            )
        else:
            print('SGD')
            aux_optimizer = optim.SGD(
                (params_dict[n] for n in sorted(aux_parameters)),
                lr=args.aux_learning_rate,
                momentum=0.9,
                weight_decay=0.0
            )
    else:
        aux_optimizer = None
    return optimizer, aux_optimizer


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def save_checkpoint(state, is_best, out_dir, filename='last_checkpoint.pth.tar'):
    
    torch.save(state, f'{out_dir}/{filename}')
    if is_best:
        name_best = filename.replace('.pth.tar','') + '_best.pth.tar'
        shutil.copyfile(f'{out_dir}/{filename}', f'{out_dir}/{name_best}')



class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)



def get_dataloaders(class_type = 'natural', dataset_path = '/data', patch_size = (256,256), batch_size = 64, val_batch_size = None, num_workers = 4, device = 'cuda', test_dir = None):
    if val_batch_size is None:
        val_batch_size = batch_size
        
    if class_type == 'natural':
        train_transforms = transforms.Compose(
            [SquarePad(patch_size), transforms.RandomCrop(patch_size), transforms.ToTensor()]
        )

        train_dataset = ImageFolder(dataset_path, split="train", transform=train_transforms)

        # print(f'total len ds: {len(train_dataset)}')
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,[0.9, 0.1])
        print(f'train len ds: {len(train_dataset)}')
        print(f'val len ds: {len(valid_dataset)}')

        # Kodak test set
        test_dataset = TestKodakDataset(data_dir = test_dir)

    else:
        raise NotImplementedError(f'Class type {class_type} not implemented yet')
    
                                        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset, 
        shuffle=False, 
        batch_size=1, 
        pin_memory=(device == "cuda"), 
        num_workers= num_workers 
    )

    print(f'Training Dataloader: {len(train_dataloader)}')

    return train_dataloader, val_dataloader, test_dataloader
    


