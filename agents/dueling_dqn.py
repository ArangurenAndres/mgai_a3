import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (14,28) → (7,14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (7,14) → (4,7)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),            # (4,7) → (2,5)
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        out = self.feature(x)
        return int(np.prod(out.size()[1:]))  # exclude batch dim


    def forward(self, x):
        x = x.float() / 255.0  # normalize
        features = self.feature(x)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

