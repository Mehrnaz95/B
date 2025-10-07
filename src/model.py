"""
model.py

Defines the main model classes for BACE.
This file mirrors the architecture used in the paper but without dataset dependency.
"""
import torch
import torch.nn as nn

class DummyBACEModel(nn.Module):
    def __init__(self, n=8, d_hidden=32):
        super().__init__()
        self.enc = nn.GRU(input_size=n, hidden_size=d_hidden, batch_first=True)
        self.fc = nn.Linear(d_hidden, n)
    def forward(self, x):
        z, _ = self.enc(x)
        return self.fc(z[:, -1, :])
