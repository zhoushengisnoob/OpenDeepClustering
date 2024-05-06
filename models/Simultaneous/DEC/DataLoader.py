# -*- coding: utf-8 -*-
"""
This is a program for data transformation part of DEC.
(Unsupervised Deep Embedding for Clustering Analysis - https://proceedings.mlr.press/v48/xieb16.pdf)
Author: Guanbao Liang
License: BSD 2 clause
"""

from torchvision import transforms


def base_transforms(mean=None, std=None, resize=None):
    """
    Function that provides basic data transformation.

    Parameters
    ----------
    mean : tuple[int]
        The mean value of normalization transformation.
    std : tuple[int]
        The variance of normalization transformation.
    resize : tuple[int]
        The image size after transformation.

    Returns
    -------
    trans : transforms.Compose
        The data transformation.
    """
    trans = []
    if resize:
        trans.append(transforms.Resize(resize))
    trans.append(transforms.ToTensor())
    if mean and std:
        trans.append(transforms.Normalize(mean=mean, std=std))
    trans = transforms.Compose(trans)
    return trans
