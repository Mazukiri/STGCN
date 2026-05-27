import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from dataset import WLASLDataset
from torch.utils.data import DataLoader
from models.baseline_lstm import BaselineLSTM

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, "..", "data", "processed")
metadata_path = os.path.join(processed_dir, "wlasl_100_metadata.csv")
landmarks_dir = os.path.join(processed_dir, "landmarks")

ds_kwargs = dict(metadata_path=metadata_path, landmarks_dir=landmarks_dir,
                 target_frames=60, return_graph=False)
loader_kwargs = dict(batch_size=32, num_workers=0, pin_memory=False)

val_loader  = DataLoader(WLASLDataset(split='val',  augment=False, **ds_kwargs), shuffle=False, **loader_kwargs)
test_loader = DataLoader(WLASLDataset(split='test', augment=False, **ds_kwargs), shuffle=False, **loader_kwargs)

model = BaselineLSTM().to(device)
model.load_state_dict(torch.load(os.path.join(script_dir, "weights", "best_lstm_model.pth"), map_location=device))
model.eval()

for name, loader in [("Val", val_loader), ("Test", test_loader)]:
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)  # inputs: (B, 60, 225)
            _, pred = out.max(1)
            correct1 += pred.eq(labels).sum().item()
            _, top5 = out.topk(5, dim=1)
            correct5 += top5.eq(labels.view(-1, 1).expand_as(top5)).any(dim=1).sum().item()
            total += labels.size(0)
    print(f"{name}: Top-1={100*correct1/total:.2f}%  Top-5={100*correct5/total:.2f}%  (n={total})")
