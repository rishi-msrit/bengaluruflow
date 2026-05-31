import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM sequence autoencoder for anomaly detection.
    Trains to reconstruct normal traffic windows.
    High reconstruction error at inference = anomaly.
    """

    def __init__(self, num_features, hidden_size=64, bottleneck_dim=32, window_size=7):
        super().__init__()

        self.num_features = num_features
        self.window_size = window_size

        # encoder
        self.encoder_lstm = nn.LSTM(num_features, hidden_size, num_layers=1, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_size, bottleneck_dim)

        # decoder
        self.decoder_fc = nn.Linear(bottleneck_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, num_features, num_layers=1, batch_first=True)

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        return self.encoder_fc(hidden[-1])

    def decode(self, latent):
        expanded = self.decoder_fc(latent)
        repeated = expanded.unsqueeze(1).repeat(1, self.window_size, 1)
        reconstruction, _ = self.decoder_lstm(repeated)
        return reconstruction

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)
