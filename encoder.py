import torch
from torch import nn 
from sublayer import MultiheadAttention, PositionwiseFeedForward
from inputprocessing import Embedding, PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, h=8, d_model=512, d_hidden=2048, dropout=0.1, *args, **kwargs) -> None:
        super(EncoderLayer, self).__init__(*args, **kwargs)
        self.attention = MultiheadAttention(num_heads=h, d_model=d_model)
        self.dropout0 = nn.Dropout(dropout)
        self.layer_norm0 = nn.LayerNorm(d_model, eps=1e-06)
        self.pwff = PositionwiseFeedForward(d_model, d_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, x, mask):
        residual0 = x

        x, attn = self.attention(x, x, x, mask)
        x = self.dropout0(x)
        x += residual0
        x = self.layer_norm0(x)
        
        residual1 = x
        
        x = self.pwff(x)
        x = self.dropout1(x)
        x += residual1
        x = self.layer_norm1(x)

        return x, attn

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_layers=6, d_model=512, num_heads=8, d_hidden=2048, dropout=0.1, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        
        self.input_layer = nn.Sequential(
            Embedding(vocab_size, d_model),
            PositionalEncoding(d_model, dropout=dropout)
        )
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(num_heads, d_model, d_hidden, dropout) for i in range(n_layers)]
        )

    def forward(self, x, mask):
        x = self.input_layer(x)

        for layer in self.encoder_layers:
            x, _ = layer(x, mask)
        return x



