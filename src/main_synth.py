"""
main_synth.py

Self-contained synthetic demo for the BACE model.
This version runs without any external data and can be used for reproducibility checks.
"""

import numpy as np
import torch
import torch.nn as nn

class SimpleGraphModel(nn.Module):
    def __init__(self, n=8, d=16):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, n) * 0.1)
        self.fc = nn.Linear(n, n)
    def forward(self, x):
        return self.fc(torch.matmul(x, self.W))

def main():
    torch.manual_seed(0)
    x = torch.randn(10, 8)              # synthetic input
    model = SimpleGraphModel()
    y = model(x)
    print("Synthetic forward pass complete. Output shape:", y.shape)

if __name__ == "__main__":
    main()
