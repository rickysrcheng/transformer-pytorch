import torch
from torch import nn 
from sublayer import MultiheadAttention, PositionwiseFeedForward
from inputprocessing import Embedding, PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, h=8, d_model=512, d_hidden=2048, dropout=0.1, *args, **kwargs) -> None:
        super(DecoderLayer, self).__init__(*args, **kwargs)
        self.rotary = kwargs.get('rotary', False)
        # Mased Attention
        self.masked_attention = MultiheadAttention(num_heads=h, d_model=d_model, rotary=self.rotary)
        self.dropout0 = nn.Dropout(dropout)
        self.layer_norm0 = nn.LayerNorm(d_model, eps=1e-06)

        self.attention = MultiheadAttention(num_heads=h, d_model=d_model, rotary=self.rotary)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-06)

        self.pwff = PositionwiseFeedForward(d_model, d_hidden)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, x, x_mask, enc, enc_mask, rotary):
        residual0 = x.clone()

        x, attn = self.masked_attention(x, x, x, x_mask)
        x = self.dropout0(x)
        x += residual0
        x = self.layer_norm0(x)
        
        residual1 = x.clone()

        # Paper's input order is V, K, Q
        # had me wondering what I did wrong cz I assumed Q, K, V... smh
        x, attn = self.attention(x, enc, enc, enc_mask)
        x = self.dropout1(x)
        x += residual1
        x = self.layer_norm1(x)

        residual2 = x.clone()

        x = self.pwff(x)
        x = self.dropout2(x)
        x += residual2
        x = self.layer_norm2(x)

        return x, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_layers=6, d_model=512, num_heads=8, d_hidden=2048, dropout=0.1, *args, **kwargs) -> None:
        super(Decoder, self).__init__(*args, **kwargs)
        
        # self.input_layer = nn.Sequential(
        #     Embedding(vocab_size, d_model),
        #     PositionalEncoding(d_model, dropout=dropout)
        # )
        self.embedding = Embedding(vocab_size, d_model)
        self.encoding = PositionalEncoding(d_model, dropout)
        self.rotary = kwargs.get('rotary', False)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(num_heads, d_model, d_hidden, dropout, rotary=self.rotary) for i in range(n_layers)]
        )

    def forward(self, x, x_mask, enc, enc_mask):
        x = self.embedding(x)
        if not self.rotary:
            x = self.encoding(x)
        for layer in self.decoder_layers:
            x, _ = layer(x, x_mask, enc, enc_mask)
        return x

