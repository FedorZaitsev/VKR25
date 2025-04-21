import os
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=MAX_LENGTH):
        
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        self.linear = nn.Linear(d_model, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        src = src.permute(1, 0, 2)
        output = self.transformer(src, src_mask, src_key_padding_mask)
        output = output.permute(1, 0, 2)
        logits = self.linear(output)
        return logits

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def train_epoch(self, loader, optimizer, criterion, scheduler=None):
        device = next(self.parameters()).device
        optimizer.zero_grad()
        
        total_loss = 0.0
        self.train()
    
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            sz = x.shape[1] - 1
            mask = self.generate_square_subsequent_mask(sz).to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = self.forward(x[:, :-1], mask)
                assert not torch.any(torch.isnan(logits))
    
                loss = criterion(logits.transpose(1, 2), y[:, 1:].long())
                assert not torch.isnan(loss)
                
                loss = loss / ACCUM_STEPS
    
            loss.backward()
    
                  
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step() 
                
            
            total_loss += loss.item() * ACCUM_STEPS
    
        if (batch_idx + 1) % ACCUM_STEPS:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Gradient clipping
    
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    
        
        return total_loss / len(loader)
    
    
    def validate(self, loader, criterion):
        device = next(self.parameters()).device
    
        total_loss = 0.0
        self.eval()
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loader):
                x = x.to(device)
                y = y.to(device)
                
                sz = x.shape[1] - 1
                mask = self.generate_square_subsequent_mask(sz).to(device)
                
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = self.forward(x[:, :-1], mask)
                    assert not torch.any(torch.isnan(logits))
        
                    loss = criterion(logits.transpose(1, 2), y[:, 1:].long())
                    assert not torch.isnan(loss)
                    
                    loss = loss
        
                total_loss += loss.item()
    
        return total_loss / len(loader)
