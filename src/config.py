from pathlib import Path

DATA = {
    'corrupted': Path('../data/corrupted_emnist'),
    'corrected': Path('../data/corrected_emnist'),
    'preds': Path('../data/preds'),
    'trues': Path('../data/trues'),
}

MODEL = {
    'embedding_dim': 20,
    'image_size': 28, # quadratic pictures only
    'vae': Path('../models/vae.pth'),
}