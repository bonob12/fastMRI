import logging
logging.disable(logging.WARNING)

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import math
from pathlib import Path
import copy
import deepspeed
import wandb

from torchinfo import summary
from tqdm import tqdm
from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet


def train_epoch(model, current_epoch_step, data_loader, lr_scheduler, loss_type):
    model.train()
    start_epoch = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    with tqdm(total=len_loader, desc=f"{'Train':<6}", ncols=120) as pbar:
        for iter, data in enumerate(data_loader):
            mask, kspace, target, maximum, _, _ = data
            mask = mask.cuda(non_blocking=True)
            kspace = kspace.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)

            output = model(kspace, mask)
            loss = loss_type(output, target, maximum)
            model.backward(loss)
            model.step()

            total_loss += loss.item()
            pbar.update(1)
            
            if iter % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4g}")

            if iter % 100 == 0:
                wandb.log({
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step = current_epoch_step + iter
                )

    return total_loss / len_loader, time.perf_counter() - start_epoch


def validate(model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()
    len_loader = len(data_loader)

    with torch.no_grad():
        with tqdm(total=len_loader, desc=f"{'Val':<6}", ncols=120) as pbar:
            for iter, data in enumerate(data_loader):
                mask, kspace, target, _, fnames, slices = data
                kspace = kspace.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                output = model(kspace, mask)

                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                    targets[fnames[i]][int(slices[i])] = target[i].numpy()
                pbar.update(1)

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

        # artifact = wandb.Artifact(name="best_model", type="model")
        # artifact.add_file(str(exp_dir / 'best_model.pt'))
        # logged_artifact = wandb.log_artifact(artifact)
        # logged_artifact.wait()

def get_optimizer_grouped_parameters(model, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "bias" in name or any(norm_name in name.lower() for norm_name in ["layernorm", "batchnorm", "instancenorm"]):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def custom_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # cosine annealing
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / optimizer.defaults['lr'], cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
def train(args):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    print('Current cuda device: ', torch.cuda.current_device())

    model = VarNet(
        num_cascades=args.cascade, 
        chans=args.chans, 
        pools=args.pools,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
    )
    model.to(device=device)

    ds_config = {
        "train_batch_size": args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps, 
        "gradient_clipping": 1.0, 
        "fp16": {
            "enabled": False,
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "contiguous_gradients": True,
        },
        "activation_checkpointing": {
            "partition_activations": False,
            "contiguous_memory_optimization": False,
            "cpu_checkpointing": False,
        },
        "compile_config": {
            "offload_activation": False,
        },
        "memory_breakdown": False,
        "wall_clock_breakdown": False,
    }

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    sample = next(iter(train_loader))
    kspace = sample[1]
    mask = sample[0]
    steps_per_epoch = len(train_loader)

    param_groups = get_optimizer_grouped_parameters(model, weight_decay=1e-4)
    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(param_groups, lr=args.lr)
    lr_scheduler = custom_lr_scheduler(
        optimizer, 
        warmup_steps=steps_per_epoch, 
        total_steps=args.num_epochs*steps_per_epoch,
        min_lr=1e-6
    )

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
    )

    summary(model.module, input_size=[tuple(kspace.shape), tuple(mask.shape)], device='cuda')

    loss_type = SSIMLoss().to(device=device)

    best_val_loss = 1.
    start_epoch = 0
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch [{epoch + 1:2d}/{args.num_epochs:2d}] ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(model, epoch * steps_per_epoch, train_loader, lr_scheduler, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(model, val_loader)
       
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        np.save(os.path.join(args.val_loss_dir, "val_loss_log"), val_loss_log)

        train_loss = torch.tensor(train_loss)
        val_loss = torch.tensor(val_loss)
        num_subjects = torch.tensor(num_subjects)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss.item(),
                "val_loss": val_loss.item(),
            },
            step = (epoch + 1) * steps_per_epoch
        )

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(f"{'TrainLoss':<10}: {train_loss:9.4g}   {'ValLoss':<8}: {val_loss:8.4g}")
        print(f"{'TrainTime':<10}: {train_time:8.2f}s   {'ValTime':<8}: {val_time:7.2f}s")

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )

    wandb.finish()
