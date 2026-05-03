import torch
import torch.nn as nn
from models.graph import Graph

class SpatialGraphConv(nn.Module):
    """Lớp Tích chập Không gian trên Đồ thị"""
    def __init__(self, in_channels, out_channels):
        super(SpatialGraphConv, self).__init__()
        # Conv2d với kernel_size 1x1 hoạt động như một lớp Linear trên số channel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        
    def forward(self, x, A):
        # x shape: (Batch, Channels, Frames, Nodes)
        # A shape: (Nodes, Nodes)
        x = self.conv(x)
        
        # Nhân ma trận dọc theo chiều Nodes (n) bằng einsum
        # b: batch, c: channels, f: frames, n: nodes, m: nodes in A
        x = torch.einsum('bcfn,nm->bcfm', x, A)
        return x

class STGCNBlock(nn.Module):
    """Một khối cấu trúc ST-GCN tiêu chuẩn"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(STGCNBlock, self).__init__()
        
        # 1. GCN: Học mối liên kết không gian giữa các khớp xương (Spatial)
        self.sgcn = SpatialGraphConv(in_channels, out_channels)
        
        # 2. TCN: Học chuyển động của khớp xương qua thời gian (Temporal)
        # Kernel=9 cho phép nhìn bao quát 9 khung hình liền nhau
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), 
                             stride=(stride, 1), padding=(4, 0))
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)
        
        # Kết nối phần dư (Residual Connection) giúp gradient lan truyền tốt hơn
        if in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x, A):
        res = self.residual(x)
        x = self.sgcn(x, A)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.bn(x)
        x += res # Cộng residual
        x = self.relu(x)
        x = self.dropout(x)
        return x

class STGCN(nn.Module):
    """Mạng ST-GCN hoàn chỉnh cho bộ dữ liệu WLASL"""
    def __init__(self, in_channels=3, num_classes=100):
        super(STGCN, self).__init__()
        
        # Khởi tạo ma trận liên kết từ lớp Graph
        self.graph = Graph()
        # Đăng ký ma trận A như một buffer của mô hình để tự động đẩy lên GPU
        A = torch.tensor(self.graph.A, dtype=torch.float32)
        self.register_buffer('A', A)
        
        # Chuẩn hóa dữ liệu đầu vào (Batch Norm trên Không gian x Kênh)
        self.data_bn = nn.BatchNorm1d(in_channels * 75)
        
        # Xếp chồng các khối STGCN
        self.layer1 = STGCNBlock(in_channels, 64)
        self.layer2 = STGCNBlock(64, 128, stride=2) # stride=2 giúp giảm một nửa số lượng frames
        self.layer3 = STGCNBlock(128, 256, stride=2)
        
        # Lớp phân loại cuối cùng
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape đầu vào: (Batch, Channels=3, Frames=60, Nodes=75)
        N, C, T, V = x.size()
        
        # Chuẩn hóa đầu vào
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        # Lan truyền qua mạng đồ thị
        x = self.layer1(x, self.A)
        x = self.layer2(x, self.A)
        x = self.layer3(x, self.A)
        
        # Global Average Pooling: Trung bình hóa mọi dữ liệu trên chiều Không gian (V) và Thời gian (T)
        # Để nén lại thành 1 vector đặc trưng duy nhất (kích thước 256) cho mỗi video
        x = x.mean(dim=-1).mean(dim=-1)
        
        # Dự đoán phân loại
        out = self.fc(x)
        return out
