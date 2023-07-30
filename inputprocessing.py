import torch
from torch import nn
import math

# used The Annotated Transformer from harvardnlp
# http://nlp.seas.harvard.edu/annotated-transformer/
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, *args, **kwargs) -> None:
        super(Embedding, self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.sqrt_d_model = math.sqrt(d_model)
        
    def forward(self, x):
        return self.embedding(x) * self.sqrt_d_model