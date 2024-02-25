"""Script containing the configuration of the project.
"""
from pathlib import Path

DATA = {
    'corrupted': Path('../data/corrupted_emnist'),
    'corrected': Path('../data/corrected_emnist.npy'),
    'preds': Path('../data/preds'),
    'trues': Path('../data/trues'),
    'masks': Path('../data/masks'),
    'scores': Path('../data/scores'),
    'cond': Path('../data/condition'),
}

MODEL = {
    'embedding_dim': 20,
    'image_size': 28, # quadratic pictures only
    'vae': Path('../models/vae.pth'),
}

EVALUATION = {
    'cond': 0.15, # threshold for defining an image as anomalous
}