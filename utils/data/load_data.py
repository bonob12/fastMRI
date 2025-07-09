import h5py
import random
import torch
import numpy as np
import pickle

from utils.data.transforms import FastmriDataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from functools import partial
from typing import Callable, Optional


def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


class FastmriSliceData(Dataset):
    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        volume_sample_rate: Optional[float] = None,
        image_cache_file: Path = Path("image_cache.pkl"),
        kspace_cache_file: Path = Path("kspace_cache.pkl"),
        num_adj_slices: int = 1,
        input_key: str = "kspace",
        target_key: str = "image_label",
        forward: bool = False,
    ):
        
        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd in SliceDataset"
        self.num_adj_slices = num_adj_slices
        self.start_adj, self.end_adj = -(self.num_adj_slices//2), self.num_adj_slices//2+1

        self.image_cache_file = image_cache_file
        self.kspace_cache_file = kspace_cache_file

        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        if self.image_cache_file.exists() and use_dataset_cache:
            with open(self.image_cache_file, "rb") as f:
                image_cache = pickle.load(f)
        else:
            image_cache = {}
        
        if self.kspace_cache_file.exists() and use_dataset_cache:
            with open(self.kspace_cache_file, "rb") as f:
                kspace_cache = pickle.load(f)
        else:
            kspace_cache = {}

        if image_cache.get(root) is None or not use_dataset_cache:
            if not forward:
                image_files = list(Path(root / "image").iterdir())
                for fname in sorted(image_files):
                    num_slices = self._get_metadata(fname)

                    self.image_examples += [
                        (fname, slice_ind) for slice_ind in range(num_slices)
                    ]
            if image_cache.get(root) is None and use_dataset_cache:
                image_cache[root] = self.image_examples
                with open(self.image_cache_file, "wb") as cache_f:
                    pickle.dump(image_cache, cache_f)
        else:
            self.image_examples = image_cache[root]
        
        if kspace_cache.get(root) is None or not use_dataset_cache:
            kspace_files = list(Path(root / "kspace").iterdir())
            for fname in sorted(kspace_files):
                num_slices = self._get_metadata(fname)

                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
            if kspace_cache.get(root) is None and use_dataset_cache:
                kspace_cache[root] = self.kspace_examples
                with open(self.kspace_cache_file, "wb") as cache_f:
                    pickle.dump(kspace_cache, cache_f)
        else:
            self.kspace_examples = kspace_cache[root]

        if volume_sample_rate < 1.0:
            vol_names = sorted(list(set([f[0].stem for f in self.kspace_examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.kspace_examples = [
                kspace_example
                for kspace_example in self.kspace_examples
                if kspace_example[0].stem in sampled_vols
            ]
            self.image_examples = [
                image_example
                for image_example in self.image_examples
                if image_example[0].stem in sampled_vols
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices
    
    def __len__(self):
        return len(self.kspace_examples)
    
    def _get_frames_indices(self, data_slice, num_slices):
        z_list = [min(max(i+data_slice, 0), num_slices-1)
                  for i in range(self.start_adj, self.end_adj)]
        return z_list
    
    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        input = []
        with h5py.File(kspace_fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
            slice_idx_list = self._get_frames_indices(dataslice, num_slices)
            for slice_idx in slice_idx_list:
                input.append(hf["kspace"][slice_idx])
            input = np.concatenate(input, axis=0)
            mask =  np.array(hf["mask"])
            
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = FastmriSliceData(
        root=data_path,
        transform=FastmriDataTransform(isforward, max_key_),
        use_dataset_cache=(args.volume_sample_rate==1.0),
        image_cache_file=data_path/"image_cache.pkl",
        kspace_cache_file=data_path/"kspace_cache.pkl",
        volume_sample_rate=args.volume_sample_rate,
        num_adj_slices=args.num_adj_slices if hasattr(args, 'num_adj_slices') else 1,
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward
    )

    worker_init = partial(worker_init_fn, seed=args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    return data_loader
