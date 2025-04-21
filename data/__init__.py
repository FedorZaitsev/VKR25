import os

SEED = int(os.getenv('SEED', 228))

BOS_TOKEN = int(os.getenv('BOS_TOKEN', 4096))
EOS_TOKEN = int(os.getenv('EOS_TOKEN', 4097))
INP_PAD_TOKEN = int(os.getenv('INP_PAD_TOKEN', 4098))
TAR_PAD_TOKEN = int(os.getenv('TAR_PAD_TOKEN', -100))
VOCAB_SIZE = int(os.getenv('VOCAB_SIZE', 4099))

MAX_LENGTH = int(os.getenv('MAX_LENGTH', 64))
OVERLAP = int(os.getenv('OVERLAP', 16))

NUM_WORKERS = int(os.getenv('NUM_WORKERS', 4))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 16))