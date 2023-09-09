import torch
import torch.nn.functional as F
from transformer import Transformer
from dataset import get_transformation
from mask import generate_padding_mask, generate_target_mask
from torchtext.datasets import multi30k, Multi30k, IWSLT2017, iwslt2017
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"


PAD_IDX, BOS_IDX, EOS_IDX = 1, 2, 3

transform, vocab = get_transformation()

@torch.no_grad()
def greedy_decoding(input_tensor, model):
    #print(vocab[SRC_LANGUAGE].lookup_tokens(list(input_tensor)))
    input_tensor = input_tensor.unsqueeze(0).long().to('cuda')
    enc_mask = generate_padding_mask(input_tensor).squeeze(-2).to('cuda')
    enc = model.encode(input_tensor, enc_mask)
    x = torch.tensor([BOS_IDX]).unsqueeze(0).long().to('cuda')
    for i in range(20):
        dec_mask = generate_target_mask(x).to('cuda')
        dec = model.decode(x, dec_mask, enc, enc_mask)
        out = model.generate(dec[:, -1]) # choose the last output to use for generation
        prob, word_tensor = torch.max(out, dim=1)
        word_tensor = word_tensor.data[0]
        x = torch.cat([x, torch.zeros(1, 1).type_as(input_tensor.data).fill_(word_tensor)], dim=1).long().to('cuda')
        if word_tensor == EOS_IDX:
            break
    return x.squeeze(0)

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
d_model = 512
n_heads = 8
d_hidden = 2048
n_layers = 6
dropout = 0.1


PATH = "./model/devout-oath-81/model-300.pt"

n_src_vocab = len(vocab[SRC_LANGUAGE])
n_tgt_vocab = len(vocab[TGT_LANGUAGE])

model = Transformer(n_src_vocab, n_tgt_vocab, 
                    d_model, n_heads, d_hidden, 
                    n_layers, dropout)
model.load_state_dict(torch.load(PATH))
model.to('cuda')
model.eval()
# string = "Wie bereits einige meiner vorredner anmerkten, wird 2009 ein besonderes Jahr sein."

# tensor = transform['de'](string)
# print(vocab[SRC_LANGUAGE].lookup_tokens(list(tensor)))
# greedy_decoding(tensor, model)

val_iter = Multi30k(root="../data", split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))



for idx, (de_st, en_st) in enumerate(val_iter):
    print("------------------------------")
    print("German:", de_st)
    x = greedy_decoding(transform['de'](de_st.rstrip("\n")), model)
    translated = vocab[TGT_LANGUAGE].lookup_tokens(list(x))[1:-1]
    print("English:", en_st)
    print("Transformer:", " ".join(translated))
    xx = transform[TGT_LANGUAGE](en_st)
    xx = torch.cat((xx, torch.ones(5))).unsqueeze(0)

    if idx > 10:
        break