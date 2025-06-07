import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class mlp(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_t, t):
        B, L, _ = x_t.shape
        t_expanded = t.unsqueeze(2).expand(-1, L, 1)  # (N_copies, L, 1)
        x_in = torch.cat([x_t, t_expanded], dim=-1)   # (N_copies, L, 4)
        score = self.net(x_in)                        # (N_copies, L, 3)
        return score