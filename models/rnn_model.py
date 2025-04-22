import os
import torch
import torch.nn as nn
import random
import numpy as np
from models import MAX_LENGTH, VOCAB_SIZE, ACCUM_STEPS


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

    def train_epoch(self, loader, optimizer, criterion, scheduler, logger=None):
        device = next(self.parameters()).device
        optimizer.zero_grad()
        
        total_loss = 0.0
        self.train()

        rand_eval_batch = random.randint(0, len(loader)-1)

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = self(x[:, :-1])
                assert not torch.any(torch.isnan(logits))

                loss = criterion(logits.transpose(1, 2), y[:, 1:].long())
                assert not torch.isnan(loss)
                
                loss = loss / ACCUM_STEPS

            loss.backward()

                  
            if (batch_idx + 1) % ACCUM_STEPS == 0:

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Gradient clipping

                if logger is not None:
                    logger.log('Train loss', loss.item())
                    if batch_idx == rand_eval_batch:
                        grads = []
                        for param in self.parameters():
                            grads.append(param.grad.view(-1).norm())
                        grads = torch.cat(grads)
                        logger.log('Mean grad norm', grads.mean())
                        logger.log('Median grad norm', grads.median())
                        logger.log('Min grad norm', grads.min())
                        logger.log('Max grad norm', grads.max())

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
        
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = self(x[:, :-1])
                    assert not torch.any(torch.isnan(logits))
        
                    loss = criterion(logits.transpose(1, 2), y[:, 1:].long())
                    assert not torch.isnan(loss)
                    
                    loss = loss

                total_loss += loss.item()

        if logger is not None:
            logger.log('Valid loss', total_loss / len(loader))
        return total_loss / len(loader)