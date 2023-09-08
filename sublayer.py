import torch
from torch import nn 
import torch.nn.functional as F
import numpy as np


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads=8, d_model=512, *args, **kwargs) -> None:
        super(MultiheadAttention, self).__init__(*args, **kwargs)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = self.d_model//self.h

        # used h separate Linear layers before realizing I can use a giant one
        # and then resize the output
        self.Q_linear = nn.Linear(self.d_model, self.h * self.d_k, bias=False)
        self.K_linear = nn.Linear(self.d_model, self.h * self.d_k, bias=False)
        self.V_linear = nn.Linear(self.d_model, self.h * self.d_k, bias=False)

        self.out  = nn.Linear(self.d_model, self.d_model)

    def ScaledDotProductAttention(self, Q, K, V, mask=None):
        score = torch.matmul(Q, torch.transpose(K, -2, -1))
        score = score/np.sqrt(self.d_k)

        # how do i mask????
        # from http://nlp.seas.harvard.edu/annotated-transformer/
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e-15)
        
        
        attention_score = F.softmax(score, dim=-1)
        output = torch.matmul(attention_score, V)
        return output, attention_score

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = q.size(0)
        len_q = q.size(1)

        len_k = k.size(1)
        len_v = v.size(1)

        # input dim is batch_size x seq_len x d_model
        Q = self.Q_linear(q)
        K = self.K_linear(k)
        V = self.V_linear(v)

        # resize to batch_size x seq_len x h x d_k
        # resize for scale dot product attention
        Q = Q.view(batch_size, len_q, self.h, self.d_k)
        K = K.view(batch_size, len_k, self.h, self.d_k)
        V = V.view(batch_size, len_v, self.h, self.d_k)
        

        # transpose to batch_size x h x seq_len x d_k
        # this way we apply attention to all heads
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        Z, attn = self.ScaledDotProductAttention(Q, K, V, mask)

        # transpose back
        Z = Z.transpose(1, 2)

        # resize back
        Z = Z.reshape(batch_size, len_q, self.d_model)
        
        Z = self.out(Z)
        return Z, attn
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_hidden=2048, *args, **kwargs) -> None:
        super(PositionwiseFeedForward, self).__init__(*args, **kwargs)
        self.layer = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )
    
    def forward(self, x):
        return self.layer(x)