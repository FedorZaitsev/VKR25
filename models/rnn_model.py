import os
import torch
import torch.nn as nn
import random
import numpy as np
from models import MAX_LENGTH, VOCAB_SIZE


class RNNModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH, embed_size=256, hidden_size=256, rnn_type=nn.LSTM, rnn_layers=1):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=embed_size)

        self.embed_ln = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(0.1)
        
        self.rnn = rnn_type(input_size=embed_size, 
                            hidden_size=hidden_size, 
                            num_layers=rnn_layers, 
                            batch_first=True)

        self.rnn_ln = nn.LayerNorm(hidden_size)
        
        self.linear = nn.Linear(hidden_size, VOCAB_SIZE)
        
        
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        
        for name, param in self.named_parameters():
            if 'weight_hh' in name:  # Recurrent weights
                torch.nn.init.orthogonal_(param)
            elif 'weight_ih' in name:  # Input weights
                torch.nn.init.xavier_normal_(param)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
                # LSTM forget gate bias = 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        nn.init.xavier_uniform(self.linear.weight)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_ln(x)
        x = self.embed_dropout(x)
        x, _ = self.rnn(x)
        x = self.rnn_ln(x)
        return self.linear(x)
