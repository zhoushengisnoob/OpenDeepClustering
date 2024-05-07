# -*- coding: utf-8 -*-
"""
This is a program for representation part of IDEC.
(Improved Deep Embedded Clustering with Local Structure Preservation)
Author: Guanbao Liang
License: BSD 2 clause
"""

from torch import nn


class StackedAutoEncoder(nn.Module):
    """
    This is a model that produces latent features from input features.
    (Improved Deep Embedded Clustering with Local Structure Preservation)

    Parameters
    ----------
    in_dim : int
        The feature dimension of the input data.
    dims : list[int]
        The numbers list of units in stacked autoencoder.
    """

    def __init__(self, in_dim, dims=None):
        super(StackedAutoEncoder, self).__init__()
        self.in_dim = in_dim
        self.dims = dims if dims else [500, 500, 2000, 10]
        self.encoders = []
        self.decoders = []
        self.initialize()

    def initialize(self):
        """
        Function that initializes the model structure.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        n_input = self.in_dim
        for i, units in enumerate(self.dims, 1):
            encoder_activation = nn.Identity() if i == len(self.dims) else nn.ReLU()
            encoder = nn.Linear(n_input, units)
            nn.init.normal_(encoder.weight, mean=0, std=0.01)
            self.encoders.append(encoder)
            self.encoders.append(encoder_activation)

            decoder_activation = nn.Identity() if i == 1 else nn.ReLU()
            decoder = nn.Linear(units, n_input)
            nn.init.normal_(decoder.weight, mean=0, std=0.01)
            self.decoders.append(decoder_activation)
            self.decoders.append(decoder)

            n_input = units

        self.encoder = nn.Sequential(*self.encoders)
        self.decoders.reverse()
        self.decoder = nn.Sequential(*self.decoders)
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x):
        """
        Forward Propagation.

        Parameters
        ----------
        x : torch.Tensor
            The images.

        Returns
        -------
        encoded : torch.Tensor
            The encoded features.
        decoded : torch.Tensor
            The reconstructed input.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
