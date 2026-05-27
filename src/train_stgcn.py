import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataset import get_dataloaders
from models.stgcn import STGCN


def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def top5_accuracy(outputs, labels):
    _, top5 = outputs.topk(5, dim=1)
    correct = top5.eq(labels.view(-1, 1).expand_as(top5))
    return correct.any(dim=1).float().sum().item()


def run_epoch(model, loader, criterion, optimizer, device, is_train,
              use_mixup=False, scaler=None, use_amp=False):
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    if use_mixup:
                        inputs, y_a, y_b, lam = mixup_batch(inputs, labels)
                        outputs = model(inputs)
                        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                        acc_labels = y_a
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        acc_labels = labels

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            else:
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                acc_labels = labels

            bs = inputs.size(0)
            total_loss += loss.item() * bs
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(acc_labels).sum().item()
            correct_top5 += top5_accuracy(outputs, acc_labels)
            total += bs

    return total_loss / total, 100.0 * correct_top1 / total, 100.0 * correct_top5 / total


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "processed"))
    metadata_path = os.path.join(processed_dir, "wlasl_100_metadata.csv")
    landmarks_dir = os.path.join(processed_dir, "landmarks")
    weights_dir = os.path.join(script_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        metadata_path,
        landmarks_dir,
        batch_size=args.batch_size,
        target_frames=60,
        return_graph=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"[*] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Workers: {args.num_workers}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    model = STGCN(in_channels=3, num_classes=100).to(device)

    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("[*] torch.compile enabled")
        except Exception:
            pass

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = -1.0
    patience_counter = 0
    history = {k: [] for k in ["train_loss", "train_acc", "train_top5",
                                "val_loss", "val_acc", "val_top5", "lr"]}

    print(f"[*] Training for up to {args.epochs} epochs (patience={args.patience})\n")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_loss, train_acc, train_top5 = run_epoch(
            model, train_loader, criterion, optimizer, device,
            is_train=True, use_mixup=True, scaler=scaler, use_amp=use_amp,
        )
        val_loss, val_acc, val_top5 = run_epoch(
            model, val_loader, criterion, optimizer, device,
            is_train=False, scaler=scaler, use_amp=use_amp,
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_top5"].append(train_top5)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5"].append(val_top5)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Loss {train_loss:.4f}/{val_loss:.4f} | "
            f"Top-1 {train_acc:.1f}%/{val_acc:.1f}% | "
            f"Top-5 {train_top5:.1f}%/{val_top5:.1f}% | "
            f"LR {current_lr:.2e}",
            flush=True,
        )

        # Save last checkpoint every epoch
        torch.save(model.state_dict(), os.path.join(weights_dir, "last_stgcn_model.pth"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_stgcn_model.pth"))
            print(f"    -> New best val acc: {best_val_acc:.2f}%", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[*] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)", flush=True)
                break

    print(f"\n[*] Done. Best val top-1: {best_val_acc:.2f}%")

    with open(os.path.join(weights_dir, "history_stgcn.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"[*] History saved to {weights_dir}/history_stgcn.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ST-GCN on WLASL-100")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--num-workers", type=int,   default=4)
    train(parser.parse_args())
