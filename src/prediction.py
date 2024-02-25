import torch 
import numpy as np

from config import DATA, MODEL
from preprocessing import preprocessing


if __name__ == '__main__':
    print("Start prediction")

    # Load model
    model = torch.load(MODEL['vae'])
    # Load data
    dataloader = preprocessing()

    # get the reconstructed images for the data
    model.eval()
    outputs = []
    inputs = []

    for batch_idx, data in enumerate(dataloader):
        pred = model(data)
        outputs.append(pred[2])
        inputs.append(data)
        if batch_idx % 100 == 0:
            print(
                f"""Predicted {batch_idx} batches: {batch_idx * len(data)}/{len(dataloader.dataset)} """
                f"""({100. * batch_idx / len(dataloader)}%) done"""
            )
    print("Finished prediction")

    # save preds and trues as torch tensors
    trues = torch.cat(inputs, dim=0)
    preds = torch.cat(outputs, dim=0)

    # save trues and preds as torch tensors
    torch.save(trues, DATA['trues'])
    torch.save(preds, DATA['preds'])

    # save preds to numpy arrays
    preds = preds.squeeze(1)
    preds_np = preds.cpu().detach().numpy()
    preds_np = preds_np = (preds_np * 255).astype(np.uint8)

    # save the reconstructed images
    np.save(DATA['corrected'], preds_np)