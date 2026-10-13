import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Architecture: Linear(128) → LSTM(128) → LSTM(128) → Linear(128) → Policy head
    
    Enhancements:
    - Dropout layers for regularization
    - ReLU activation between linear layers
    - Proper initialization
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_actions: int = 3, dropout: float = 0.1):
        super().__init__()

        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout_input = nn.Dropout(dropout)

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        # Policy head (BUY / HOLD / SELL)
        self.policy_head = nn.Linear(hidden_dim, num_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for better training stability.
        Xavier/Glorot initialization for linear layers.
        """
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: shape (batch, lookback, features)
            
        Returns:
            logits: shape (batch, num_actions)
        """
        # Linear projection per timestep
        x = self.input_proj(x)
        x = self.dropout_input(x)

        # LSTM stack
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Use last timestep output
        x = x[:, -1, :]

        # Fully connected layer with activation
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # Policy head (no activation - raw logits for cross-entropy)
        logits = self.policy_head(x)

        return logits
