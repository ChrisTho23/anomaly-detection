import torch

from config import DATA

if __name__ == "__main__":
    preds = torch.load(DATA["preds"])
    trues = torch.load(DATA["trues"])

    # generate masks, i.e. absolute diff of each pixel to original image
    masks = []

    for pred, true in zip(preds, trues):
        masks.append(abs(true-pred))

    masks = torch.cat(masks, dim=0)
