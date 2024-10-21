


import torch
import torch.nn as nn
import numpy as np
import math
from .layers import PositionalEncoding, EncoderLayer, DecoderLayer



class Decoder(nn.Module):
    def __init__(self, configs, n_layers):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(hidden_size=configs.hidden_size,
                                                  ffn_hidden=configs.hidden_size,
                                                  n_head=configs.num_heads,
                                                  drop_prob=configs.drop_rate)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(configs.hidden_size, configs.hidden_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output

class Encoder(nn.Module):

    def __init__(self, configs, n_layers):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(hidden_size=configs.hidden_size,
                                                  ffn_hidden=configs.hidden_size,
                                                  n_head=configs.num_heads,
                                                  drop_prob=configs.drop_rate)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask=None):
        for layer in self.layers:
            x = layer(x, s_mask)

        return x