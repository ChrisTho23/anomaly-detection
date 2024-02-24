from pathlib import Path

DATA = {
    'corrupted': Path('../data/corrupted_emnist'),
}

MODEL = {
    'embedding_dim': 20,
    'image_size': 28, # quadratic pictures only
}