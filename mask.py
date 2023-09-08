import torch

PAD_IDX = 1

def generate_padding_mask(tensor):
    return (tensor != PAD_IDX).unsqueeze(-2)

def generate_target_mask(tensor):
    # returns a 1 x sz x sz mask
    padding_mask = generate_padding_mask(tensor)
    sz = tensor.size(-1)
    future_mask = (torch.triu(torch.ones((sz, sz)) == 1)).transpose(0, 1).unsqueeze(0)
    #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return padding_mask & future_mask.type_as(padding_mask.data)
    