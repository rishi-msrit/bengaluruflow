import torch
import torch.nn as nn


class TransformerForecaster(nn.Module):
    """
    Small Transformer encoder for traffic forecasting.
    Implemented from scratch using nn.TransformerEncoder.
    """

    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, window_size=7,
                 horizon=3, num_targets=3):
        super().__init__()

        self.horizon = horizon
        self.num_targets = num_targets
        self.window_size = window_size

        self.input_projection = nn.Linear(input_size, d_model)
        # learned positional embedding for each time step
        self.positional_embedding = nn.Embedding(window_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon * num_targets)

    def forward(self, x):
        # x: (batch, T, input_size)
        _, seq_len, _ = x.shape

        x = self.input_projection(x)  # (batch, T, d_model)

        positions = torch.arange(seq_len, device=x.device)
        x = x + self.positional_embedding(positions).unsqueeze(0)

        x = self.transformer_encoder(x)  # (batch, T, d_model)
        x = x.mean(dim=1)               # mean pool over time

        out = self.fc(x)
        return out.view(-1, self.horizon, self.num_targets)
