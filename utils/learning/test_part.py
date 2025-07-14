import numpy as np
import torch
import importlib

from collections import defaultdict
from argparse import Namespace
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders

def resolve_class(class_path: str):
    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not resolve class '{class_path}'. Error: {e}")

def test(model, data_loader):
    model['cnn'].eval()
    for task in ['brain', 'knee']:
        for acc in ['acc4', 'acc8']:
            model[task][acc].eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, image, accs, fnames) in data_loader:
            mask = mask.cuda(non_blocking=True)
            kspace = kspace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)

            image = image.squeeze(0)
            image = image.unsqueeze(1)
            output = model['cnn'](image)

            task = ['brain', 'knee']
            task = task[torch.argmax(output, dim=1).item()]

            for slice_idx in range(kspace.shape[1]):
                sliced_kspace = kspace[:, slice_idx]
                output = model[task][accs[0]](sliced_kspace, mask)
                reconstructions[fnames[0]][slice_idx] = output[0].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions


def forward(args):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    print ('Current cuda device ', torch.cuda.current_device())

    checkpoint = {
        'cnn': torch.load(args.cnn_checkpoint, map_location=device, weights_only=False),
        'brain': {
            'acc4': torch.load(args.brain_acc4_checkpoint, map_location=device, weights_only=False),
            'acc8': torch.load(args.brain_acc8_checkpoint, map_location=device, weights_only=False),
        },
        'knee': {
            'acc4': torch.load(args.knee_acc4_checkpoint, map_location=device, weights_only=False),
            'acc8': torch.load(args.knee_acc8_checkpoint, map_location=device, weights_only=False),
        },
    }

    model = defaultdict(dict)
    model['cnn'] = resolve_class(checkpoint['cnn']['args'].model_name)().to(device=device)

    for task in ['brain', 'knee']:
        for acc in ['acc4', 'acc8']:
            saved_args = checkpoint[task][acc]['args']
            model_name = saved_args.model_name
            ModelClass = resolve_class(model_name)
            if model_name.endswith('VarNet'):
                model_instance = ModelClass(
                    num_cascades=saved_args.cascade, 
                    chans=saved_args.chans, 
                    pools=saved_args.pools,
                    sens_chans=saved_args.sens_chans,
                    sens_pools=saved_args.sens_pools,
                ).to(device=device)
            elif model_name.endswith('PromptMR'):
                model_instance = ModelClass(
                    num_cascades=saved_args.num_cascades,
                    num_adj_slices=saved_args.num_adj_slices,
                    n_feat0=saved_args.n_feat0,
                    feature_dim=saved_args.feature_dim,
                    prompt_dim=saved_args.prompt_dim,
                    sens_n_feat0=saved_args.sens_n_feat0,
                    sens_feature_dim=saved_args.sens_feature_dim,
                    sens_prompt_dim=saved_args.sens_prompt_dim,
                    len_prompt=saved_args.len_prompt,
                    prompt_size=saved_args.prompt_size,
                    n_enc_cab=saved_args.n_enc_cab,
                    n_dec_cab=saved_args.n_dec_cab,
                    n_skip_cab=saved_args.n_skip_cab,
                    n_bottleneck_cab=saved_args.n_bottleneck_cab,
                    n_buffer=saved_args.n_buffer,
                    n_history=saved_args.n_history,
                    no_use_ca=saved_args.no_use_ca,
                    learnable_prompt=saved_args.learnable_prompt,
                    adaptive_input=saved_args.adaptive_input,
                    use_sens_adj=saved_args.use_sens_adj,
                    compute_sens_per_coil=saved_args.compute_sens_per_coil,
                ).to(device=device)
            else:
                raise ValueError("No matching model")
            model[task][acc]=model_instance
    
    print(f"cnn: epoch={checkpoint['cnn']['epoch']}")
    for task in ['brain', 'knee']:
        for acc in ['acc4', 'acc8']:
            print(f"{task}_{acc}: epoch={checkpoint[task][acc]['epoch']}")
            model[task][acc].load_state_dict(checkpoint[task][acc]['model'])
    
    forward_loader = create_data_loaders(data_path=args.data_path, args=Namespace(seed=430), data_type='test', slicedata='TestSliceData')
    reconstructions = test(model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=None)