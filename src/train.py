import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WLASLDataset
from models.baseline_lstm import BaselineLSTM

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[*] Training on device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "processed"))
    metadata_path = os.path.join(processed_dir, "wlasl_100_metadata.csv")
    landmarks_dir = os.path.join(processed_dir, "landmarks")
    
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    TARGET_FRAMES = 60
    NUM_CLASSES = 100

    ds_kwargs = dict(metadata_path=metadata_path, landmarks_dir=landmarks_dir,
                     target_frames=TARGET_FRAMES, return_graph=False)
    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    train_loader = DataLoader(WLASLDataset(split='train', augment=True, **ds_kwargs),
                              shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(WLASLDataset(split='val',   augment=False, **ds_kwargs),
                              shuffle=False, **loader_kwargs)
    
    model = BaselineLSTM(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    os.makedirs(os.path.join(script_dir, "weights"), exist_ok=True)
    best_model_path = os.path.join(script_dir, "weights", "best_lstm_model.pth")
    
    # [PHASE 3] Lưu lịch sử train để vẽ biểu đồ
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("[*] Starting LSTM training loop...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        train_loss = train_loss / total_train
        train_acc = 100. * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / total_val
        val_acc = 100. * correct_val / total_val
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
              
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    -> Saved new best model with Val Acc: {best_val_acc:.2f}%")
            
    print(f"[*] Training finished! Best Validation Accuracy: {best_val_acc:.2f}%")
    
    history_path = os.path.join(script_dir, "weights", "history_lstm.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
    print(f"[*] Saved LSTM training history to: {history_path}")

if __name__ == '__main__':
    train()
