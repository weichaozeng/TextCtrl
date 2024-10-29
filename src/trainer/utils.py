import os
import json
import torch
import argparse
import importlib
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.utilities import rank_zero_only

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def module_requires_grad(module: nn.Module):
    for param in module.parameters():
        if not param.requires_grad:
            return False
    return True

@rank_zero_only
def pl_on_train_tart(pl_module: pl.LightningModule):
    wandb_logger = pl_module.logger.experiment
    if isinstance(wandb_logger, WandbLogger):
        print("Logging code")
        wandb_logger.log_code(
            os.getcwd(), 
            include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".yaml") 
        )
    elif isinstance(wandb_logger, TensorBoardLogger):
        print("Logging git info")
        wandb_logger.log_hyperparams({"git_version": os.popen("git log -1").read().split("\n")[0]})
        
    print("***** Start training *****")
    num_samples = len(pl_module.trainer.train_dataloader.dataset)
    max_epoch = pl_module.trainer.max_epochs
    total_step = pl_module.trainer.estimated_stepping_batches
    total_batch_size = round(num_samples * max_epoch / total_step)
    print(f"  Num examples = {num_samples}")
    print(f"  Num Epochs = {max_epoch}")
    print(f"  Total GPU device number: {pl_module.trainer.num_devices}")
    print(f"  Gradient Accumulation steps = {pl_module.trainer.accumulate_grad_batches}")
    print(f"  Instant batch size: {round(total_batch_size * pl_module.trainer.num_devices)}") 
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Total optimization steps = {total_step}")
    