# -*- coding: utf-8 -*-
"""
This is a program for seeding random numbers to ensure reproducibility.
Author: Guanbao Liang
License: BSD 2 clause
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed=2024):
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
