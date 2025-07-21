import logging
logging.disable(logging.WARNING)

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import datetime
import math
from pathlib import Path
import deepspeed
import wandb
import importlib
import cv2 

from tqdm import tqdm
from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, seed_fix
from utils.common.loss_function import SSIMLoss

def resolve_class(class_path: str):
    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not resolve class '{class_path}'. Error: {e}")

def train_epoch(model_engine, epoch, data_loader, lr_scheduler, loss_type, slicedata):
    model_engine.train()
    start_epoch = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    with tqdm(total=len_loader, desc=f"{'Train':<6}", ncols=120) as pbar:
        for iter, data in enumerate(data_loader):
            if slicedata == 'FastmriSliceData':
                mask, kspace, target, maximum, fnames, _ = data
                mask = mask.cuda(non_blocking=True)
                kspace = kspace.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                maximum = maximum.cuda(non_blocking=True)
                output = model_engine(kspace, mask)

                binary_masks = []

                for i in range(output.shape[0]):
                    target_np = target[i].cpu().numpy()

                    binary_mask = np.zeros(target_np.shape)
                    if 'knee' in str(fnames[i]):
                        binary_mask[target_np>2e-5] = 1
                    elif 'brain' in str(fnames[i]):
                        binary_mask[target_np>5e-5] = 1
                    
                    kernel = np.ones((3, 3), np.uint8)
                    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
                    binary_mask = cv2.dilate(binary_mask, kernel, iterations=15)
                    binary_mask = cv2.erode(binary_mask, kernel, iterations=14)  
                    binary_mask = (torch.from_numpy(binary_mask).to(device=output.device)).type(torch.float)
                    binary_masks.append(binary_mask)
                
                binary_masks = torch.stack(binary_masks).to(device=output.device)
                loss = loss_type(output*binary_masks, target*binary_masks, maximum)
            else:
                image, target = data
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                image = image.squeeze(0)
                image = image.unsqueeze(1)
                output = model_engine(image)
                loss = loss_type(output, target)
            
            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            pbar.update(1)
            
            if iter % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4g}")

            if iter % 50 == 0:
                wandb.log({
                        "epoch": epoch + 1,
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step = epoch * len_loader + iter
                )

    return total_loss / len_loader, time.perf_counter() - start_epoch

def validate(model_engine, data_loader, loss_type, slicedata):
    model_engine.eval()
    start = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    incorret = 0

    with torch.no_grad():
        with tqdm(total=len_loader, desc=f"{'Val':<6}", ncols=120) as pbar:
            for iter, data in enumerate(data_loader):
                if slicedata == 'FastmriSliceData':
                    mask, kspace, target, maximum, fnames, _ = data
                    mask = mask.cuda(non_blocking=True)
                    kspace = kspace.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    maximum = maximum.cuda(non_blocking=True)
                    output = model_engine(kspace, mask)

                    binary_masks = []

                    for i in range(output.shape[0]):
                        target_np = target[i].cpu().numpy()

                        binary_mask = np.zeros(target_np.shape)
                        if 'knee' in str(fnames[i]):
                            binary_mask[target_np>2e-5] = 1
                        elif 'brain' in str(fnames[i]):
                            binary_mask[target_np>5e-5] = 1
                        
                        kernel = np.ones((3, 3), np.uint8)
                        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
                        binary_mask = cv2.dilate(binary_mask, kernel, iterations=15)
                        binary_mask = cv2.erode(binary_mask, kernel, iterations=14)  
                        binary_mask = (torch.from_numpy(binary_mask).to(device=output.device)).type(torch.float)
                        binary_masks.append(binary_mask)
            
                    binary_masks = torch.stack(binary_masks).to(device=output.device)
                    loss = loss_type(output*binary_masks, target*binary_masks, maximum)
                else:
                    image, target = data
                    image = image.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                    image = image.squeeze(0)
                    image = image.unsqueeze(1)
                    output = model_engine(image)
                    loss = loss_type(output, target)

                    pred = torch.argmax(output, dim=1)
                    if pred != target:
                        incorret += 1

                total_loss += loss.item()
                pbar.update(1)
    
    if slicedata == 'FastmriSliceData':
        return total_loss / len_loader, time.perf_counter() - start
    else:
        return total_loss / len_loader, 1 - incorret / len_loader, time.perf_counter() - start


def save_model(args, epoch, model_engine, save_artifact):
    client_state = {'epoch': epoch}
    model_engine.save_checkpoint(args.exp_dir, tag=f"epoch-{epoch}", client_state=client_state)
    if save_artifact:
        artifact = wandb.Artifact(name=f'epoch-{epoch}-model', type="model")
        artifact.add_dir(str(args.exp_dir / f"epoch-{epoch}"))
        wandb.log_artifact(artifact)

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
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / optimizer.defaults['lr'], cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(args):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    print('Current cuda device: ', torch.cuda.current_device())

    wandb.init(
        project=str(args.net_name),    
        dir=f"../result/{args.net_name}",
        config=vars(args),
    )

    seed_fix(args.seed)
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    wandb.run.name = run_name = f"{timestamp}-{wandb.run.id}"

    args.exp_dir = Path('../result') / args.net_name / 'checkpoints' / run_name
    args.loss_log_dir = Path('../result') / args.net_name / 'loss_log' / run_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.loss_log_dir.mkdir(parents=True, exist_ok=True)

    ModelClass = resolve_class(args.model_name)

    if args.model_name.endswith('VarNet'):
        model = ModelClass(
            num_cascades=args.cascade, 
            chans=args.chans, 
            pools=args.pools,
            sens_chans=args.sens_chans,
            sens_pools=args.sens_pools,
        )
        loss_type = SSIMLoss().to(device=device)
        slicedata = 'FastmriSliceData'
    elif args.model_name.endswith('PromptMR'):
        model = ModelClass(
            num_cascades=args.num_cascades,
            num_adj_slices=args.num_adj_slices,
            n_feat0=args.n_feat0,
            feature_dim=args.feature_dim,
            prompt_dim=args.prompt_dim,
            sens_n_feat0=args.sens_n_feat0,
            sens_feature_dim=args.sens_feature_dim,
            sens_prompt_dim=args.sens_prompt_dim,
            len_prompt=args.len_prompt,
            prompt_size=args.prompt_size,
            n_enc_cab=args.n_enc_cab,
            n_dec_cab=args.n_dec_cab,
            n_skip_cab=args.n_skip_cab,
            n_bottleneck_cab=args.n_bottleneck_cab,
            n_buffer=args.n_buffer,
            n_history=args.n_history,
            no_use_ca=args.no_use_ca,
            learnable_prompt=args.learnable_prompt,
            adaptive_input=args.adaptive_input,
            use_sens_adj=args.use_sens_adj,
            compute_sens_per_coil=args.compute_sens_per_coil,
        )
        loss_type = SSIMLoss().to(device=device)
        slicedata = 'FastmriSliceData'
    elif args.model_name.endswith('CNN'):
        model = ModelClass()
        loss_type = nn.CrossEntropyLoss().to(device=device)
        slicedata = 'CNNSliceData'
    else:
        raise ValueError("No matching model")

    model.to(device=device)

    ds_config = {
        "train_batch_size": args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps, 
        "gradient_clipping": 0.01, 
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
            "cpu_checkpointing": True,
        },
        "memory_breakdown": False,
        "wall_clock_breakdown": False,
    }

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True, data_type='train', slicedata=slicedata)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, data_type='val', slicedata=slicedata)

    steps_per_epoch = len(train_loader)

    param_groups = get_optimizer_grouped_parameters(model, weight_decay=1e-4)
    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(param_groups, lr=args.lr)

    optimizer_steps_per_epoch = steps_per_epoch // args.gradient_accumulation_steps
    lr_scheduler = custom_lr_scheduler(
        optimizer, 
        warmup_steps=args.warmup_epochs*optimizer_steps_per_epoch, 
        total_steps=args.num_epochs*optimizer_steps_per_epoch,
        min_lr = 0,
    )

    if args.init_from_cascade is not None:
        state_dict = torch.load(args.init_from_cascade/"mp_rank_00_model_states.pt", map_location='cpu')
        cascade_dict = {
            k.replace('module.cascades.0.', ''): v
            for k, v in state_dict['module'].items()
            if k.startswith('module.cascades.0.')
        }

        for i in range(args.num_cascades):
            model.cascades[i].load_state_dict(cascade_dict, strict=False)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
    )
    
    if args.restart_from_checkpoint is not None:
        _, client_state = model_engine.load_checkpoint(
            args.restart_from_checkpoint.parent, 
            tag=args.restart_from_checkpoint.name, 
            load_optimizer_states=True, 
            load_lr_scheduler_states=args.continue_lr_scheduler
        )
        start_epoch = client_state.get('epoch', 0)
    else:
        start_epoch = 0

    loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch [{epoch + 1:2d}/{args.num_epochs:2d}] ............... {args.net_name} ...............')
        
        train_loader.dataset.update_epoch(epoch + 1)
        train_loss, train_time = train_epoch(model_engine, epoch, train_loader, lr_scheduler, loss_type, slicedata)
        
        if slicedata == 'FastmriSliceData':
            val_loss, val_time = validate(model_engine, val_loader, loss_type, slicedata)
        else:
            val_loss, accuracy, val_time = validate(model_engine, val_loader, loss_type, slicedata)

        loss_log = np.append(loss_log, np.array([[epoch, train_loss]]), axis=0)
        np.save(os.path.join(args.loss_log_dir, "loss_log"), loss_log)

        save_model(args, epoch + 1, model_engine, args.save_artifact)
        print(f"{'TrainLoss':<10}: {train_loss:9.4g}   {'ValLoss':<8}: {val_loss:8.4g}")
        print(f"{'TrainTime':<10}: {train_time:8.2f}s   {'ValTime':<8}: {val_time:7.2f}s")

        if slicedata != 'FastmriSliceData':
            print(f"{'Accuracy':<10}: {accuracy:9.4g}")
            wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": accuracy,
                },
                step = (epoch + 1) * steps_per_epoch - 1
            )
        else:
            wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                step = (epoch + 1) * steps_per_epoch - 1
            )

    if wandb.run is not None:
        wandb.finish()

