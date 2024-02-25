"""Finds anomalies in the predictions based on mask and pre-defined threshold.
"""
import torch

from config import DATA, EVALUATION

if __name__ == "__main__":
    # load torch tensors of predictions and true images (shape: (126300, 1, 28, 28))
    preds = torch.load(DATA["preds"]) 
    trues = torch.load(DATA["trues"])

    # generate masks, i.e. list of tensros containing 
    # absolute diff of each pixel to original image
    masks = []

    for pred, true in zip(preds, trues):
        masks.append(abs(true-pred))

    masks = torch.cat(masks, dim=0) # shape: (126300, 28, 28)
    torch.save(masks, DATA["masks"])

    # sum the values of anomalous pixels of all pixels in an image
    # and weigh that by the number of pixels as evaluation metrics

    scores = (masks.sum(1).sum(1) / (28 * 28)).detach().numpy()
    torch.save(scores, DATA["scores"])

    cond = scores > EVALUATION["cond"]
    torch.save(cond, DATA["cond"])

    print(f"Number of images with anomalies: {cond.sum()}")
