from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
import torch
from torch import nn
from transformer import Transformer

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

## turns out figuring out how to feed the dataset is the hardest part of this exercise lol
## Data handling referenced: https://pytorch.org/tutorials/beginner/translation_transformer.html

## Transformation Steps: 
## 1. Tokenize the raw string
## 2. Create Vocabulary, and numericalize
## 3. Tensorize and add <bos> and <eos> tokens

# Dataloader: 
# Collate and preprocess raw string batch using the above transformations
# then pad the batch

# Mask:
# Create mask using upper triangular function
# Would also need to mask out the padding

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

TOKENIZER_LANGUAGE = {'de': "de_core_news_sm", 
                      'en': "en_core_web_sm"}

def get_transformation():

    ### 1. Tokenizer
    token_transform = {}
    token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language=f"{TOKENIZER_LANGUAGE[SRC_LANGUAGE]}")
    token_transform[TGT_LANGUAGE] = get_tokenizer("spacy", language=f"{TOKENIZER_LANGUAGE[TGT_LANGUAGE]}")


    ### 2. Create Vocab class & numericalize tokens
    def yield_tokens(data_iter: Iterable, language: str, token_transform : dict):
        ln_idx = {'de': 0, 'en': 1}
        idx = ln_idx[language]
        for data in data_iter:
            yield token_transform[language](data[idx])
        
    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    
    # Create vocab object and numericalize
    vocab_transform = {}
    train_iter = Multi30k(root="../data", split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    vocab_transform[SRC_LANGUAGE] = build_vocab_from_iterator(yield_tokens(train_iter, SRC_LANGUAGE, token_transform),
                                                              min_freq=1, specials=special_symbols, special_first=True)
    vocab_transform[SRC_LANGUAGE].set_default_index(UNK_IDX)

    train_iter = Multi30k(root="../data", split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    vocab_transform[TGT_LANGUAGE] = build_vocab_from_iterator(yield_tokens(train_iter, TGT_LANGUAGE, token_transform),
                                                              min_freq=1, specials=special_symbols, special_first=True)
    vocab_transform[TGT_LANGUAGE].set_default_index(UNK_IDX)

    ### 3. Tensorize
    def tensor_transform(input_vect: List[int]):
        return torch.cat((
            torch.tensor([BOS_IDX]),
            torch.tensor(input_vect),
            torch.tensor([EOS_IDX])
        ))
    
    def transform(*transforms):
        def apply_transform(text):
            ret = text
            for transform in transforms:
                ret = transform(ret)
            return ret
        return apply_transform

    text_transform = {}

    text_transform[SRC_LANGUAGE] = transform(token_transform[SRC_LANGUAGE], 
                                             vocab_transform[SRC_LANGUAGE],
                                             tensor_transform)
    text_transform[TGT_LANGUAGE] = transform(token_transform[TGT_LANGUAGE], 
                                             vocab_transform[TGT_LANGUAGE],
                                             tensor_transform)



    return text_transform, vocab_transform

