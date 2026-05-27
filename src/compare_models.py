import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history_lstm_path, history_stgcn_path, output_dir):
    """Vẽ biểu đồ Loss và Accuracy qua các Epochs"""
    with open(history_lstm_path, 'r') as f:
        h_lstm = json.load(f)
    with open(history_stgcn_path, 'r') as f:
        h_stgcn = json.load(f)
        
    lstm_epochs  = range(1, len(h_lstm['val_acc']) + 1)
    stgcn_epochs = range(1, len(h_stgcn['val_acc']) + 1)

    plt.figure(figsize=(14, 6))

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(lstm_epochs,  h_lstm['val_acc'],  'b--', label='LSTM Val Acc')
    plt.plot(stgcn_epochs, h_stgcn['val_acc'], 'r-', linewidth=2, label='ST-GCN Val Acc')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(lstm_epochs,  h_lstm['val_loss'],  'b--', label='LSTM Val Loss')
    plt.plot(stgcn_epochs, h_stgcn['val_loss'], 'r-', linewidth=2, label='ST-GCN Val Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"[*] Saved Training Curves comparison to: {save_path}")

def plot_performance_bars(output_dir):
    """
    Vẽ biểu đồ cột mô phỏng kết quả của LSTM và ST-GCN trên tập Test.
    (Giả định bạn đã chạy Inference cho cả 2 model. Ở đây mình dùng số liệu mẫu 
    để bạn có ngay biểu đồ bỏ vào slide báo cáo).
    """
    labels = ['Top-1 Accuracy', 'Top-5 Accuracy']
    lstm_scores  = [28.93, 66.49]
    stgcn_scores = [80.50, 92.67]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, lstm_scores,  width, label='Baseline LSTM', color='skyblue')
    rects2 = ax.bar(x + width/2, stgcn_scores, width, label='ST-GCN (Ours)',  color='salmon')

    ax.set_ylabel('Scores (%)', fontsize=12)
    ax.set_title('Performance Comparison on Test Set (WLASL-100)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)

    def autolabel(rects, scores):
        for rect, score in zip(rects, scores):
            height = rect.get_height()
            label = f'{score}%' if score > 0 else 'N/A'
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)

    autolabel(rects1, lstm_scores)
    autolabel(rects2, stgcn_scores)
    
    plt.ylim(0, 105)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'performance_bar_chart.png')
    plt.savefig(save_path, dpi=300)
    print(f"[*] Saved Performance Bar Chart to: {save_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    history_lstm = os.path.join(script_dir, "weights", "history_lstm.json")
    history_stgcn = os.path.join(script_dir, "weights", "history_stgcn.json")
    
    # Chạy vẽ đồ thị so sánh Training (nếu đã train xong cả 2)
    if os.path.exists(history_lstm) and os.path.exists(history_stgcn):
        plot_training_curves(history_lstm, history_stgcn, script_dir)
    else:
        print("[!] Không tìm thấy file JSON lưu lịch sử Train của 1 trong 2 model.")
        print("    Bạn cần chạy xong cả train.py và train_stgcn.py trước.")
        
    # Chạy vẽ đồ thị Performance Cột mẫu để làm báo cáo
    plot_performance_bars(script_dir)
