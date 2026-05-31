import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, horizon=3, num_targets=3):
        super().__init__()

        self.horizon = horizon
        self.num_targets = num_targets

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, horizon * num_targets)

    def forward(self, x):
        # x: (batch, T, input_size)
        _, (hidden, _) = self.lstm(x)
        # take last layer's hidden state
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out.view(-1, self.horizon, self.num_targets)
