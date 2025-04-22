import os

VOCAB_SIZE = int(os.getenv('VOCAB_SIZE', 4099))
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 64))

ACCUM_STEPS = int(os.getenv('ACCUM_STEPS', 1))