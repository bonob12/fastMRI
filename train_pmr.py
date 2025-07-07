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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--not_sweep', action='store_true', help='This is not wandb sweep call')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('--model_name', type=str, default="utils.model.promptmr.PromptMR", help='Name of model')
    parser.add_argument('--data_path_train', type=Path, default='../Data/train/', help='Directory of train data')
    parser.add_argument('--data_path_val', type=Path, default='../Data/val/', help='Directory of validation data')
    
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    parser.add_argument('--num_cascades', type=int, default=1)
    parser.add_argument('--num_adj_slices', type=int, default=1)

    parser.add_argument('--n_feat0', type=int, default=48)
    parser.add_argument('--feature_dim', nargs='+', type=int, default=[72, 96, 120])
    parser.add_argument('--prompt_dim', nargs='+', type=int, default=[24, 48, 72])

    parser.add_argument('--sens_n_feat0', type=int, default=24)
    parser.add_argument('--sens_feature_dim', nargs='+', type=int, default=[36, 48, 60])
    parser.add_argument('--sens_prompt_dim', nargs='+', type=int, default=[12, 24, 36])

    parser.add_argument('--len_prompt', nargs='+', type=int, default=[5, 5, 5])
    parser.add_argument('--prompt_size', nargs='+', type=int, default=[64, 32, 16])

    parser.add_argument('--n_enc_cab', nargs='+', type=int, default=[2, 3, 3])
    parser.add_argument('--n_dec_cab', nargs='+', type=int, default=[2, 2, 3])
    parser.add_argument('--n_skip_cab', nargs='+', type=int, default=[1, 1, 1])
    parser.add_argument('--n_bottleneck_cab', type=int, default=3)

    parser.add_argument('--n_buffer', type=int, default=0)
    parser.add_argument('--n_history', type=int, default=0)

    parser.add_argument('--no_use_ca', type=str2bool)
    parser.add_argument('--learnable_prompt', type=str2bool)
    parser.add_argument('--adaptive_input', type=str2bool)
    parser.add_argument('--use_sens_adj', type=str2bool)
    parser.add_argument('--compute_sens_per_coil', type=str2bool)

    parser.add_argument('--input_key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target_key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max_key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    parser.add_argument('--restart_from_checkpoint', type=Path, default=None)
    parser.add_argument('--continue_lr_scheduler', type=str2bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    if args.not_sweep:
        wandb.init(
            project=str(args.net_name),    
            dir=f"../result/{args.net_name}",
            config=vars(args),
        )
    else:
        wandb.init(
            dir="../results/test_promptmr_sweep"
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
