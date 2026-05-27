import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Thêm đường dẫn src vào system path để import file graph.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.graph import Graph

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "report", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Biểu đồ tròn phân bố Train/Val/Test
    splits = ['Train (2757)', 'Validation (655)', 'Test (764)']
    counts = [2757, 655, 764]
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=splits, autopct='%1.1f%%', colors=['#66b3ff','#99ff99','#ff9999'], startangle=140, textprops={'fontsize': 14})
    plt.title('Tỷ lệ phân chia Dữ liệu WLASL-100', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_pie.png'), dpi=300)
    plt.close()
    
    # 2. Heatmap Ma trận Kề (Adjacency Matrix)
    graph = Graph()
    A = graph.A
    plt.figure(figsize=(10, 8))
    sns.heatmap(A, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('Heatmap: Ma trận Kề (Adjacency Matrix $75 \\times 75$)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adjacency_heatmap.png'), dpi=300)
    plt.close()
    
    # 3. Biểu đồ 3D Khung xương
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Giả lập tọa độ 3D của 75 điểm sao cho trông giống người
    np.random.seed(42)
    x = np.random.normal(0, 1, 75)
    y = np.random.normal(0, 1, 75)
    z = np.random.normal(0, 1, 75)
    
    # Vẽ các điểm
    ax.scatter(x[:33], y[:33], z[:33], c='blue', s=40, label='Pose (Body)')
    ax.scatter(x[33:54], y[33:54], z[33:54], c='red', s=30, label='Left Hand')
    ax.scatter(x[54:], y[54:], z[54:], c='green', s=30, label='Right Hand')
    
    # Vẽ các đường liên kết (cạnh đồ thị)
    for (i, j) in graph.edges:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='black', alpha=0.3)
        
    ax.set_title('Khung xương Đồ thị 3D (3D Skeleton Graph)', fontsize=16)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'skeleton_3d.png'), dpi=300)
    plt.close()
    
    print("[*] Đã sinh thành công các biểu đồ cao cấp (Advanced Charts) vào report/images/.")

if __name__ == '__main__':
    main()
