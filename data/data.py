import os
import torch
import torch.nn as nn
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from data import BOS_TOKEN, EOS_TOKEN, INP_PAD_TOKEN, TAR_PAD_TOKEN, VOCAB_SIZE, MAX_LENGTH, OVERLAP, NUM_WORKERS, BATCH_SIZE, SEED


def read_sequences(root_dir, device='cpu'):
	filenames = []

	for root, dirs, files in os.walk(root_dir):
	    for file in files:
	        filenames.append(os.path.join(root, file))

	sequences = [None] * len(filenames)

	for i, filename in tqdm(enumerate(filenames)):
	    sequences[i] = torch.load(filename, map_location=device).permute(2, 0, 1).flatten()

	return sequences


def chunk_sequence(sequence, chunk_size=MAX_LENGTH, overlap=OVERLAP):
    chunks = []
    stride = chunk_size - overlap
    for i in range(0, len(sequence), stride):
        chunk = sequence[i:i+chunk_size-2]
        if len(chunk) > overlap:
            chunks.append(torch.cat([torch.tensor([BOS_TOKEN]), chunk.to('cpu'), torch.tensor([EOS_TOKEN])]))
    return chunks


class ChunkedMusicDataset(Dataset):
    def __init__(self, token_sequences, chunk_size=MAX_LENGTH, overlap=OVERLAP):
        self.chunks = []
        
        for seq in token_sequences:
            chunks = chunk_sequence(seq, chunk_size, overlap)
            self.chunks.extend(chunks)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk, chunk


def get_train_val_dataset(sequences, train_ratio=0.8):
	dataset = ChunkedMusicDataset(sequences)
	g = torch.Generator().manual_seed(SEED)
	return torch.utils.data.random_split(dataset, [train_ratio, 1 - train_ratio], generator=g)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=INP_PAD_TOKEN)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=TAR_PAD_TOKEN)
    return inputs_padded, targets_padded


def get_loader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn):
	def seed_worker(worker_id):
		worker_seed = torch.initial_seed() % 2**32
		np.random.seed(worker_seed)
		random.seed(worker_seed)

	g = torch.Generator().manual_seed(SEED)
	return DataLoader(dataset, 
		batch_size=BATCH_SIZE, 
		num_workers=NUM_WORKERS, 
		worker_init_fn=seed_worker,
		generator=g, 
		collate_fn=collate_fn
		)
