import torch
from torch import nn 
import torch.nn.functional as F
import numpy as np
import torchtext
import torchtext.datasets as datasets
from encoder import Encoder
from decoder import Decoder
from inputprocessing import PositionalEncoding, Embedding
from lmhead import LMHead

class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, 
                 d_model = 512, num_heads=8, d_hidden=2048, n_layers=6,
                 dropout=0.1, *args, **kwargs) -> None:
        super(Transformer, self).__init__(*args, **kwargs)

        self.d_model = d_model

        self.encoder = Encoder(vocab_size=n_src_vocab, n_layers=n_layers, 
                               d_model=d_model, num_heads=num_heads, d_hidden=d_hidden,
                               dropout=dropout)
        self.decoder = Decoder(vocab_size=n_tgt_vocab, n_layers=n_layers,
                               d_model=d_model, num_heads=num_heads, d_hidden=d_hidden,
                               dropout=dropout)
        
        self.lmhead = LMHead(d_model, n_tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.lmhead(self.decoder(tgt, tgt_mask, self.encoder(src, src_mask), src_mask))
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, tgt_mask, enc, enc_mask):
        return self.decoder(tgt, tgt_mask, enc, enc_mask)
    
    def generate(self, x):
        assert x.size(-1) == self.d_model
        return self.lmhead(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)



def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

if __name__ == "__main__":
    h = 8
    d_model = 512
    d_hidden = 2048
    b = 4
    dropout = 0.1
    seq_len = 64

    net = Transformer(11, 11)
    net.apply(init_weights)
    print(net)

    src = torch.randint(0, 11, (b, 10))
    print(src.size())
    src_mask = torch.ones(1, 10)

    memory = net.encode(src, src_mask)
    ys = torch.zeros(b, 5).type_as(src)
    print(memory.size())
    print(ys, ys.size())
    print(subsequent_mask(ys.size(1)).type_as(src.data).size())

    print("Decoding")
    out = net.decode(ys, subsequent_mask(ys.size(1)).type_as(src.data), memory, src_mask)
    print(src.size())
    print(out.size())
    print(torch.max(net.generate(out[:,-1]), dim=1)[1])
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)