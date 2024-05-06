# -*- coding: utf-8 -*-
"""
This is a program for environment initialization.
Author: Guanbao Liang
License: BSD 2 clause
"""
import argparse
import os

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import (
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnet18,
    ResNet18_Weights,
)

from utils.manual_seed import seed_everything
from utils.load_config import load_config, load_config_list


def init_seed(seed=2024):
    """
    Function sets the seed for random number generation, including torch, numpy, random, and hash.

    Parameters
    ----------
    seed : int
        The seed you input.

    Returns
    -------
    None
    """
    seed_everything(seed=seed)


def init_device(device="cpu", device_list=[]):
    """
    Function sets the running device.

    Parameters
    ----------
    device : str
        The device name, like cpu or cuda.
    device_list : List[str]
        The device list.

    Returns
    -------
    None
    """
    if device != "cpu":
        assert len(device_list) > 0, "Device List is empty."
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(device_id) for device_id in device_list
        )


def init(config_type="yaml", config_file=None, parser=None):
    """
    Function loads the configuration files, sets the seed and devices.

    Parameters
    ----------
    config_type : str
        The configuration file type.
    config_file : str
        The configuration file of configuration files.
    parser : argparse.ArgumentParser | None
        The module simplifies command-line argument handling.

    Returns
    -------
    args : Namespace
        The namespace of the arguments.
    """
    if config_file == None:
        config_file = "configs/base.yaml"
    if parser == None:
        parser = argparse.ArgumentParser()
    if type(config_file) == list:
        parser = load_config_list(config_type, config_file, parser)
    else:
        parser = load_config(config_type, config_file, parser)
    args = parser.parse_args()
    init_seed(args.seed)
    init_device(args.device, args.cuda_visible_devices)
    
    return args


def init_tb_writer(log_dir="./"):
    """
    Function creates the tensorboard writer.

    Parameters
    ----------
    log_dir : str
        The log directory.

    Returns
    -------
    writer : torch.utils.tensorboard.SummaryWriter
    """
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def init_optimizer(
    optimizer_name="adam", lr=1e-4, weight_decay=0, params=None, *args, **kwargs
):
    """
    Function creates the optimizer.

    Parameters
    ----------
    optimizer_name : name
        The optimizer that user want to use.
    lr : float
        The learning rate for the optimizer.
    weight_decay : float
        The L2 norm of the weights for the loss function.
    params : tuple[torch.Tensor]
        The training params of the model.
    args : Any
        The other positional arguments for setting the optimizer.
    kwargs : Dict
        The other keyword arguments.

    Returns
    -------
    optimizer : torch.optim.Optimizer
    """
    if optimizer_name.lower() == "adam":
        beta1 = kwargs.get("adam_beta1", 0.9)
        beta2 = kwargs.get("adam_beta2", 0.999)
        betas = (beta1, beta2)
        optimizer = optim.Adam(
            params=params, lr=lr, betas=betas, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        momentum = kwargs.get("sgd_momentum", 0.9)
        optimizer = optim.SGD(
            params=params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise NotImplementedError(
            f"{optimizer_name.lower()} optimizer is not implemented."
        )

    return optimizer


def init_backbone(backbone_name="resnet50", pretrained=True):
    """
    Function creates the vision backbone.

    Parameters
    ----------
    backbone_name : str
        The specified vision backbone.
    pretrained : bool
        Whether to choose the pretrained weights for vision backbone.

    Returns
    -------
    vis_backbone : nn.Moudule
        The vision backbone like resnet34, resnet50.
    """
    if backbone_name.lower() == "resnet50":
        vis_backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif backbone_name.lower() == "resnet34":
        vis_backbone = resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif backbone_name.lower() == "resnet18":
        vis_backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
    else:
        raise NotImplementedError("{} is not implemented!".format(backbone_name))
    return vis_backbone
