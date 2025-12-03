"""Autoencoder model for gene selection"""

import torch
from torch import nn


class FeatureScreeningAutoencoder(nn.Module):
    """Autoencoder with learnable feature importance weights"""

    def __init__(self, input_size, embedding_size, dp=0.2, lk=0.2):
        super(FeatureScreeningAutoencoder, self).__init__()

        self.feature_importance = nn.Parameter(torch.ones(input_size))

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(lk, inplace=True),
            nn.Dropout(dp),
            nn.Linear(512, 256),
            nn.LeakyReLU(lk, inplace=True),
            nn.Dropout(dp),
            nn.Linear(256, embedding_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.LeakyReLU(lk, inplace=True),
            nn.Dropout(dp),
            nn.Linear(256, 512),
            nn.LeakyReLU(lk, inplace=True),
            nn.Linear(512, input_size),
        )

    def forward(self, x):
        screened_features = x * self.feature_importance
        encoded = self.encoder(screened_features)
        decoded = self.decoder(encoded)
        return screened_features, encoded, decoded
