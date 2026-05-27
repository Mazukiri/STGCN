"""
Evaluate best ST-GCN model and print top-1, top-5, macro F1 on val and test splits.
Run from src/ directory:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python src/evaluate.py
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from dataset import get_dataloaders
from models.stgcn import STGCN


def evaluate_split(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            _, top5 = outputs.topk(5, dim=1)
            correct_top5 += top5.eq(labels.view(-1, 1).expand_as(top5)).any(dim=1).sum().item()
            total += labels.size(0)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    top1 = 100.0 * np.sum(all_preds == all_labels) / total
    top5 = 100.0 * correct_top5 / total
    f1   = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return top1, top5, f1, all_preds, all_labels


def plot_confusion_matrix(all_labels, all_preds, output_path):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False,
                linewidths=0, cbar=True)
    plt.title('Confusion Matrix — ST-GCN on WLASL-100 Test Set', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[*] Saved confusion matrix to: {output_path}")


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Device: {device}")

    script_dir    = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "processed"))
    weights_path  = os.path.join(script_dir, "weights", "best_stgcn_model.pth")

    if not os.path.exists(weights_path):
        print(f"[!] Weights not found: {weights_path}")
        return

    _, val_loader, test_loader = get_dataloaders(
        os.path.join(processed_dir, "wlasl_100_metadata.csv"),
        os.path.join(processed_dir, "landmarks"),
        batch_size=32,
        target_frames=60,
        return_graph=True,
    )

    model = STGCN(in_channels=3, num_classes=100).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    print("\n=== ST-GCN Evaluation ===")
    test_preds = test_labels = None
    for name, loader in [("Val", val_loader), ("Test", test_loader)]:
        top1, top5, f1, preds, labels = evaluate_split(model, loader, device)
        print(f"{name:4s}  Top-1: {top1:.2f}%  Top-5: {top5:.2f}%  Macro-F1: {f1:.2f}%")
        if name == "Test":
            test_preds, test_labels = preds, labels

    cm_path = os.path.join(script_dir, "confusion_matrix.png")
    plot_confusion_matrix(test_labels, test_preds, cm_path)


if __name__ == "__main__":
    main()
