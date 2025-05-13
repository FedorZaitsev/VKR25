import os
from torchph import nn as phnn
from torchph import pershom
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models import MAX_LENGTH, VOCAB_SIZE, ACCUM_STEPS

ph = pershom.pershom_backend.__C.VRCompCuda__vr_persistence_l1

INIT_RANGE=1e-3

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout)
        self.attention_weights = None  # Stores last attention weights
        
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # Copy of original code but storing attention weights
        x, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal
        )
        self.attention_weights = attn_weights
        return self.dropout1(x)

class CustomTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        self.intermediate_outputs = []
        
        for mod in self.layers:
            output = mod(output, src_mask=mask, 
                        src_key_padding_mask=src_key_padding_mask)
            self.intermediate_outputs.append(output)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=MAX_LENGTH):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TopoTransformerModel(TransformerModel):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=MAX_LENGTH, slayer_elements=256):     
        super(TopoTransformerModel, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        # self.transformer = CustomTransformerEncoder(encoder_layers, num_layers)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        self.transformer.layers[-1] = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        
        self.linear = nn.Linear(d_model, vocab_size)

        self.init_weights()
      
        # self.slayer = nn.ModuleList()
        # self.topo_linear = nn.ModuleList()
        # for i in range(num_layers):
        #     self.slayer.append(phnn.slayer.SLayerRationalHat(n_elements=slayer_elements))
        #     self.topo_linear.append(nn.Linear(slayer_elements, vocab_size))
        
        # for tl in self.topo_linear:
        #     tl.bias.data.zero_()
        #     tl.weight.data.zero_()    
            
        self.slayer = phnn.slayer.SLayerRationalHat(n_elements=slayer_elements)
        self.topo_linear = nn.Linear(slayer_elements, vocab_size)
        
        initrange=INIT_RANGE
        
        self.topo_linear.bias.data.zero_()
        # self.topo_linear.weight.data.zero_()
        self.topo_linear.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        src = src.permute(1, 0, 2)
        output = self.transformer(src, src_mask, src_key_padding_mask)
        output = output.permute(1, 0, 2)

        logits = self.linear(output)

        # topo_logits = 0
        # for i, inter_output in enumerate(self.transformer.intermediate_outputs):
        #     topo_logits += self.topo_linear[i](self.slayer[i](torch.stack([ph(src_i, 0, 0)[0][0] for src_i in inter_output.permute(1, 0, 2)])))
        
        topo_logits = self.topo_linear(torch.cat([self.slayer(ph(src_i, 0, 0)[0][0].unsqueeze(0)) for src_i in self.transformer.layers[-1].attention_weights.permute(0, 2, 1)], 0))
        
        logits = logits + topo_logits[:, None, :]
        return logits
