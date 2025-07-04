import torch
import argparse
import shutil
import datetime
import os, sys
import wandb
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--not_sweep', action='store_true', help='This is not wandb sweep call')

    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report_interval', type=int, default=10, help='Report interval')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data_path_train', type=Path, default='../Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data_path_val', type=Path, default='../Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet')
    parser.add_argument('--pools', type=int, default=4)
    parser.add_argument('--sens_pools', type=int, default=4)
    parser.add_argument('--input_key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target_key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max_key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()

    if args.not_sweep:
        wandb.init(
            project="test_varnet",    
            dir="../result/test_Varnet",
            config=vars(args),
        )
    else:
        wandb.init(
            dir="../result/test_Varnet",
        )
        config = wandb.config
        args = argparse.Namespace(**config)
        args.net_name = Path(args.net_name)
        args.data_path_train = Path(args.data_path_train)
        args.data_path_val= Path(args.data_path_val)


    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    run_id = wandb.run.id
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"{timestamp}-{run_id}"
    wandb.run.name = run_name

    args.exp_dir = Path('../result') / args.net_name / 'checkpoints' / run_name
    args.val_dir = Path('../result') / args.net_name / 'reconstructions_val' / run_name
    args.val_loss_dir = Path('../result') / args.net_name / 'val_loss_log' / run_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    args.val_loss_dir.mkdir(parents=True, exist_ok=True)

    train(args)
