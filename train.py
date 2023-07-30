from dataset import get_transformation
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
import torch
from torch import nn
from transformer import Transformer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb
from timeit import default_timer as timer

wandb.init(
    project="Transformer",

    config={
        'epochs' : 1000,
        'batch_size' : 128,
        'd_model' : 512,
        'd_hidden' : 2048,
        'n_heads' : 8,
        'n_layers' : 6,
        'dropout' : 0.1,
        'warmup_steps' : 4000
    }
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

PAD_IDX = 1

TOKENIZER_LANGUAGE = {'de': "de_core_news_sm", 
                      'en': "en_core_web_sm"}

transform, vocab = get_transformation()

def collate_fn(batch):
    # given a batch, apply transform and padding to turn it into a batch tensor
    de_batch, en_batch = [], []

    for de_string, en_string in batch:
        de_batch.append(transform['de'](de_string.rstrip("\n")))
        en_batch.append(transform['en'](en_string.rstrip("\n")))
    
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX, batch_first=True)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=True)
    return de_batch, en_batch

def generate_padding_mask(tensor):
    return (tensor != PAD_IDX).unsqueeze(-2)

def generate_target_mask(tensor):
    padding_mask = generate_padding_mask(tensor)
    sz = tensor.size(-1)
    future_mask = (torch.triu(torch.ones((sz, sz)) == 1)).transpose(0, 1).unsqueeze(0)
    #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return padding_mask & future_mask.type_as(padding_mask.data)


n_src_vocab = len(vocab[SRC_LANGUAGE])
n_tgt_vocab = len(vocab[TGT_LANGUAGE])

model = Transformer(n_src_vocab, n_tgt_vocab, 
                    wandb.config.d_model, wandb.config.n_heads, 
                    wandb.config.d_hidden, wandb.config.n_layers, 
                    wandb.config.dropout)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

model.apply(init_weights)

model.to(DEVICE)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)

def rate(step, d_model, warmup_steps):
    if step == 0:
        step = 1
    return d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

lrate = lambda step : rate(step, wandb.config.d_model, wandb.config.warmup_steps)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrate)

for e in range(wandb.config.epochs):
    model.train()
    losses = 0
    train_iter = Multi30k(root="../data", split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, wandb.config.batch_size, collate_fn=collate_fn, shuffle=True)
    epoch_time = timer()
    for src, tgt in train_dataloader:
        src_mask = generate_padding_mask(src).to(DEVICE)
        tgt_mask = generate_target_mask(tgt).to(DEVICE)
        src = src.to(DEVICE).long()
        tgt = tgt.to(DEVICE).long()

        logits = model(src, tgt, src_mask, tgt_mask)
        
        optimizer.zero_grad()

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]
        wandb.log({"learning_rate": float(lr)})
        losses += loss.item()
    
    model.eval()

    val_iter = Multi30k(root="../data", split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, wandb.config.batch_size, collate_fn=collate_fn)
    val_losses = 0
    
    for src, tgt in val_dataloader:
        src_mask = generate_padding_mask(src).to(DEVICE)
        tgt_mask = generate_target_mask(tgt).to(DEVICE)
        src = src.to(DEVICE).long()
        tgt = tgt.to(DEVICE).long()

        logits = model(src, tgt, src_mask, tgt_mask)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

        val_losses += loss.item()

    train_loss = losses/len(list(train_dataloader))
    val_loss = val_losses/len(list(val_dataloader))
    wandb.log({"epoch": e, "train_loss": train_loss, "val_loss": val_loss})
    end_time = timer()
    print(f"Epoch: {e}, train loss = {train_loss}, val loss = {val_loss}, time: {end_time - epoch_time}")

    if (e+1) % 10 == 0:
        torch.save(model.state_dict(), f"./model/model-{e+1}.pt")

