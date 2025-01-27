import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MarketAnalysisNetwork(nn.Module):
    def __init__(self, input_size=64, hidden_size=128):
        super().__init__()
        # Advanced neural architecture for market analysis
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.dropout = nn.Dropout(0.2)
        
        # Feature extraction layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x, hidden=None):
        # Process market data through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        x = F.relu(self.fc1(attn_output[:, -1, :]))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)
