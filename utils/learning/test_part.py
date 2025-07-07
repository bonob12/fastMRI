import numpy as np
import torch
import importlib

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet

def resolve_class(class_path: str):
    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not resolve class '{class_path}'. Error: {e}")

def test(model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    print ('Current cuda device ', torch.cuda.current_device())

    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    saved_args = checkpoint['args']

    ModelClass = resolve_class(saved_args.model_name)

    if "VarNet" in saved_args.model_name:
        model = ModelClass(
            num_cascades=saved_args.cascade, 
            chans=saved_args.chans, 
            pools=saved_args.pools,
            sens_chans=saved_args.sens_chans,
            sens_pools=saved_args.sens_pools,
        )
    elif "PromptMR" in saved_args.model_name:
        model = ModelClass(
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
        )
    else:
        raise ValueError("No matching model")
    
    model.to(device=device)
    
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = saved_args, isforward = True)
    reconstructions, inputs = test(model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)