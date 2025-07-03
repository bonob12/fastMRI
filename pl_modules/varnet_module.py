import torch
from data import DataTransform
from pl_modules import MriModule
from typing import List
import copy 
from mri_utils.losses import SSIMLoss
import torch.nn.functional as F
import importlib

def get_model_class(module_name, class_name="VarNet"):
    """
    Dynamically imports the specified module and retrieves the class.

    Args:
        module_name (str): The module to import (e.g., 'model.m1', 'model.m2').
        class_name (str): The class to retrieve from the module (default: 'PromptMR').

    Returns:
        type: The imported class.
    """
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

class VarnetModule(MriModule):

    def __init__(
        self,
        model_version: str = "varnet",
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        lr: float = 0.0002,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.sens_chans = sens_chans
        self.sens_pools = sens_pools
        self.chans = chans
        self.pools = pools

        self.lr = lr

        self.model_version = model_version
        Varnet = get_model_class(f"model.{model_version}")

        self.varnet = Varnet(
            num_cascades = num_cascades,
            sens_chans = sens_chans,
            sens_pools = sens_pools,
            chans = chans,
            pools = pools,
        )

        self.loss = SSIMLoss()

    def configure_optimizers(self):

        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr
        )

        return [optim]
    
    def forward(self, masked_kspace, mask):
        return self.varnet(masked_kspace, mask)
        
    
    def training_step(self, batch, batch_idx):

        mask, kspace, target, maximum, fname, slice_num = batch
        output = self(kspace, mask)

        train_loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=maximum)
        self.log("train_loss", train_loss, prog_bar=True, batch_size=kspace.size(0))

        if torch.isnan(train_loss):
            raise ValueError(f'nan loss on {fname} of slice {slice_num}')
        return train_loss

    def validation_step(self, batch, batch_idx):

        mask, kspace, target, maximum, fname, slice_num = batch
        output = self(kspace, mask)

        val_loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=maximum)
        self.log("val_loss", val_loss, prog_bar=True, batch_size=kspace.size(0))

        if torch.isnan(val_loss):
            raise ValueError(f'nan loss on {fname} of slice {slice_num}')
        return val_loss

