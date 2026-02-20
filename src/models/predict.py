import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits.squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)

    return np.array(preds)