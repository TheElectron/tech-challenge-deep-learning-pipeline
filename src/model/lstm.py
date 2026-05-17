"""LSTM architectures for multi-asset stock price prediction.

Two configurations are provided:

Config A — StackedLSTM
    Two sequential LSTM layers with dropout between them.
    The first layer extracts short-range sequential patterns; the second
    compresses them into a fixed-size representation for the output head.
    ~100 k parameters with the default sizes.

Config B — AttentionLSTM
    A 2-layer LSTM followed by multi-head self-attention and a residual
    connection.  Attention allows the model to explicitly weight which
    of the 60 input time steps matter most for the prediction, rather
    than relying solely on the LSTM hidden state.
    ~350 k parameters with the default sizes.

Recommended choice: StackedLSTM (see pipeline.py for full justification).
"""

import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    """Config A: Two stacked LSTM layers.

    Args:
        input_size:   number of features per time step (1 for univariate)
        hidden_sizes: (h1, h2) — hidden units for layer 1 and layer 2
        dropout:      dropout probability applied after each LSTM layer
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_sizes: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_size)
        out, _ = self.lstm1(x)           # (B, T, h1)
        out = self.drop1(out)
        out, _ = self.lstm2(out)         # (B, T, h2)
        out = self.drop2(out[:, -1, :])  # last time step → (B, h2)
        return self.fc(out)              # (B, 1)


class AttentionLSTM(nn.Module):
    """Config B: LSTM followed by multi-head self-attention.

    The attention layer re-weights every position in the LSTM output
    sequence before collapsing to the last time step.  A residual
    connection and LayerNorm stabilise training.

    Args:
        input_size:  number of features per time step
        hidden_size: LSTM hidden units and attention embed dim
        num_heads:   number of attention heads (must divide hidden_size)
        dropout:     dropout probability
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)                                  # (B, T, H)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (B, T, H)
        out = self.norm(lstm_out + attn_out)                        # residual
        return self.fc(out[:, -1, :])                               # (B, 1)
