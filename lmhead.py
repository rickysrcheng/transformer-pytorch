import torch
from torch import nn 

class LMHead(nn.Module):
    def __init__(self, d_model, n_output, *args, **kwargs) -> None:
        super(LMHead, self).__init__(*args, **kwargs)

        self.layer = nn.Sequential(
            nn.Linear(d_model, n_output),
            nn.LogSoftmax(-1)
        )
    
    def forward(self, x):
        return self.layer(x)