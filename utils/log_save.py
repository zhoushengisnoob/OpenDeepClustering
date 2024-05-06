# -*- coding: utf-8 -*-
"""
This is a program for saving logs and models.
Author: Guanbao Liang
License: BSD 2 clause
"""

import os
import datetime
import torch


def save_param(log_dir, param_dict, file_name="params.txt"):
    """
    Function saves the experiment parameters.

    Parameters
    ----------
    log_dir : str
        The log directory.
    param_dict : dict
        The parameters of experiment.
    file_name : str
        The file name.

    Returns
    -------
    None
    """
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)
    full_path = os.path.join(log_dir, file_name)
    with open(full_path, "w") as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[Saved at {current_time}]\n\n")
        for key, value in param_dict.items():
            f.write(f"{key}: {value}\n")
        f.flush()


def save_train_details(log_dir, train_details, file_name="train_details.txt"):
    """
    Function saves the training details like loss.

    Parameters
    ----------
    log_dir : str
        The log directory.
    train_details : str
        The training details.
    file_name : str
        The file name.

    Returns
    -------
    None
    """
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)
    full_path = os.path.join(log_dir, file_name)
    with open(full_path, "a") as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[Saved at {current_time}] ")
        f.write(train_details)
        f.flush()


def save_eva_details(log_dir, eva_details, file_name="eva_details.txt"):
    """
    Function saves the evaluation details like accuracy.

    Parameters
    ----------
    log_dir : str
        The log directory.
    eva_details : str
        The evaluation details.
    file_name : str
        The file name.

    Returns
    -------
    None
    """
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)
    full_path = os.path.join(log_dir, file_name)
    with open(full_path, "a") as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[Saved at {current_time}] ")
        f.write(eva_details)
        f.flush()


def save_model(save_model_dir, ckpt, iteration_num):
    """
    Function saves the trained models.

    Parameters
    ----------
    log_dir : str
        The log directory.
    ckpt : dict
        The trained models.
    iteration_num : int
        The num of iteration you saves the model.

    Returns
    -------
    None
    """
    if os.path.exists(save_model_dir) is False:
        os.makedirs(save_model_dir)
    full_path = os.path.join(save_model_dir, "ckpt_%d.pt" % iteration_num)
    torch.save(ckpt, full_path)
