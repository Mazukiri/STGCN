import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from dataset import get_dataloaders
from models.baseline_lstm import BaselineLSTM

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Evaluating on device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "processed"))
    metadata_path = os.path.join(processed_dir, "wlasl_100_metadata.csv")
    landmarks_dir = os.path.join(processed_dir, "landmarks")
    mapping_path = os.path.join(processed_dir, "label_mapping.json")
    best_model_path = os.path.join(script_dir, "weights", "best_lstm_model.pth")
    
    if not os.path.exists(best_model_path):
        print(f"[!] Error: Model weights not found at {best_model_path}")
        print("    Please run train.py first.")
        return
        
    with open(mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
        
    # Tạo mapping ngược từ index -> từ vựng (gloss)
    idx_to_gloss = {v: k for k, v in label_mapping.items()}
    
    # Chỉ lấy Test Loader
    _, _, test_loader = get_dataloaders(
        metadata_path, 
        landmarks_dir, 
        batch_size=32, 
        target_frames=60
    )
    
    model = BaselineLSTM(num_classes=100).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Dùng cho Top-5
    correct_top5 = 0
    total = 0
    
    print("[*] Running inference on Test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Tính Top-1
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Tính Top-5
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.cpu().numpy()
            targets = labels.cpu().numpy()
            
            for i in range(len(targets)):
                if targets[i] in top5_pred[i]:
                    correct_top5 += 1
                total += 1
                
    # Tính tổng quan
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    top1_acc = 100. * np.sum(all_preds == all_labels) / total
    top5_acc = 100. * correct_top5 / total
    
    print(f"\n=================================")
    print(f"       EVALUATION RESULTS        ")
    print(f"=================================")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    
    # In báo cáo chi tiết (Classification Report)
    print("\n[*] Generating Classification Report...")
    # Chỉ hiển thị các label có xuất hiện
    unique_labels = np.unique(all_labels)
    target_names = [idx_to_gloss[i] for i in unique_labels]
    
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    # Vẽ Confusion Matrix (lấy Top 20 từ xuất hiện nhiều nhất để vẽ cho đỡ rối)
    print("[*] Drawing Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Thu gọn CM cho đẹp: Chỉ lấy Top 20 nhãn đầu tiên
    top_20_idx = unique_labels[:20]
    cm_top20 = cm[np.ix_(top_20_idx, top_20_idx)]
    labels_top20 = [idx_to_gloss[i] for i in top_20_idx]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_top20, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_top20, yticklabels=labels_top20)
    plt.title('Confusion Matrix (Top 20 Glosses)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(script_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[*] Saved Confusion Matrix plot to: {cm_path}")

if __name__ == '__main__':
    evaluate()
