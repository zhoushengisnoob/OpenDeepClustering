# -*- coding: utf-8 -*-
"""
This is a program for clustering and loss part of IDEC.
(Improved Deep Embedded Clustering with Local Structure Preservation)
Author: Guanbao Liang
License: BSD 2 clause
"""

import torch

from torch import nn


class Clustering(nn.Module):
    """
    This is a model that calculates the probability of the sample belonging to each cluster.
    (Improved Deep Embedded Clustering with Local Structure Preservation)

    Parameters
    ----------
    in_dim : int
        The feature dimension of the input data.
    n_clusters : int
        The number of clusters.
    alpha : float
        The parameter in Student's t-distribution which defaults to 1.0.
    weights : numpy.ndarray
        The weights of centroids which is obtained by Kmeans.

    Examples
    --------
    # >>> model = Clustering(in_dim=784,n_clusters=10,alpha=1.0,weights=None)
    # >>> out = model(input_data)
    """

    def __init__(self, in_dim, n_clusters, alpha=1.0, weights=None):
        super(Clustering, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.centroids = nn.Parameter(
            torch.empty(n_clusters, in_dim), requires_grad=True
        )
        self.initial_weights = weights
        self.initialize()

    def initialize(self):
        """
        Functions that initializes the centroids.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        nn.init.xavier_uniform_(self.centroids)
        if self.initial_weights is not None:
            weights_tensor = torch.tensor(self.initial_weights).float()
            self.centroids.data = weights_tensor

    def forward(self, inputs):
        """
        Function that calculates the probability of assigning sample i to cluster j.

        Parameters
        ----------
        inputs : torch.Tensor
            The data you input.

        Returns
        -------
        q : torch.Tensor
            The data of probabilities of assigning all samples to all clusters.
        """
        q = 1.0 / (
            1.0
            + (
                torch.sum(
                    torch.square(torch.unsqueeze(inputs, dim=1) - self.centroids), dim=2
                )
                / self.alpha
            )
        )
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, dim=1), 0, 1)
        return q


class Loss(nn.Module):
    """
    This is a Loss object.

    Parameters
    ----------
    gamma : float
        The coefficient of cluster loss.
    """

    def __init__(self, gamma=1.0):
        super(Loss, self).__init__()
        self.recon_loss = nn.MSELoss()
        self.cluster_loss = nn.KLDivLoss(reduction="batchmean")
        self.gamma = gamma

    def forward(self, pred, target, recon, x):
        """
        Loss calculation.

        Parameters
        ----------
        pred : torch.Tensor
            The model predictions after logarithm.
        target : torch.Tensor
            The ground-truth labels.
        recon : torch.Tensor
            The reconstructed input.
        x : torch.Tensor
            The original input.

        Returns
        -------
        losses : torch.Tensor
            The losses calculated by loss function.
        """
        lossC = self.cluster_loss(pred, target)
        lossR = self.recon_loss(recon, x) * self.gamma
        return lossC + lossR
