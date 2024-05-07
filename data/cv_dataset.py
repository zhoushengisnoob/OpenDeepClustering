# -*- coding: utf-8 -*-
"""
This is a program for loading datasets.
Author: Guanbao Liang
License: BSD 2 clause
"""

import torchvision


def load_torchvision_dataset(dataset_name, dataset_dir, train=True, transform=None):
    """
    Function loads the torchvision dataset.

    Parameters
    ----------
    dataset_name : str
        The dataset name like mnist.
    dataset_dir : str
        The dataset will download in the dataset_dir.
    train : bool
        Whether to choose the training dataset.

    Returns
    -------
    dataset : Dataset
        Specified dataset.
    """
    if dataset_name.lower() == "mnist":
        dataset = torchvision.datasets.MNIST(
            root=dataset_dir, train=train, transform=transform, download=True
        )
    elif dataset_name.lower() == "fashionmnist":
        dataset = torchvision.datasets.FashionMNIST(
            root=dataset_dir, train=train, transform=transform, download=True
        )
    elif dataset_name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir, train=train, transform=transform, download=True
        )
    elif dataset_name.lower() == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=dataset_dir, train=train, transform=transform, download=True
        )
    elif dataset_name.lower() == "stl10":
        # modal = "train" if train else "test"
        if type(train) ==bool:
            modal = "train" if train else "test"
        elif type(train) ==str:
            modal = train
        dataset = torchvision.datasets.STL10(
            root=dataset_dir, split=modal, transform=transform, download=True
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
    return dataset
