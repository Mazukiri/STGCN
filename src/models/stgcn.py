import torch
import torch.nn as nn
from models.graph import Graph

class SpatialGraphConv(nn.Module):
    """Lớp Tích chập Không gian trên Đồ thị"""
    def __init__(self, in_channels, out_channels):
        super(SpatialGraphConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x, A):
        # x: (Batch, Channels, Frames, Nodes)
        # A: (Nodes, Nodes)
        x = self.conv(x)
        x = torch.einsum('bcfn,nm->bcfm', x, A)
        return x

class STGCNBlock(nn.Module):
    """Một khối cấu trúc ST-GCN tiêu chuẩn"""
    def __init__(self, in_channels, out_channels, num_nodes=75, stride=1, dropout_rate=0.5):
        super(STGCNBlock, self).__init__()

        # Learnable per-block graph correction (zero-init → identical to fixed A at start)
        self.A_adapt = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # 1. GCN: Spatial
        self.sgcn = SpatialGraphConv(in_channels, out_channels)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        # 2. TCN: Temporal (kernel=9, 9 consecutive frames)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1),
                             stride=(stride, 1), padding=(4, 0))

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

        if in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, A):
        A_eff = A + self.A_adapt
        res = self.residual(x)
        x = self.sgcn(x, A_eff)
        x = self.bn_gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.bn(x)
        x += res
        x = self.relu(x)
        x = self.dropout(x)
        return x

class STGCN(nn.Module):
    """Mạng ST-GCN hoàn chỉnh cho bộ dữ liệu WLASL"""
    def __init__(self, in_channels=3, num_classes=100, num_nodes=75):
        super(STGCN, self).__init__()

        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32)
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 6 blocks: (64×2) → (128×2, stride=2) → (256×2, stride=2)
        # Temporal: 60 → 60 → 30 → 30 → 15 → 15
        self.layer1 = STGCNBlock(in_channels, 64,  num_nodes=num_nodes, dropout_rate=0.1)
        self.layer2 = STGCNBlock(64,          64,  num_nodes=num_nodes, dropout_rate=0.1)
        self.layer3 = STGCNBlock(64,          128, num_nodes=num_nodes, stride=2, dropout_rate=0.25)
        self.layer4 = STGCNBlock(128,         128, num_nodes=num_nodes, dropout_rate=0.25)
        self.layer5 = STGCNBlock(128,         256, num_nodes=num_nodes, stride=2, dropout_rate=0.5)
        self.layer6 = STGCNBlock(256,         256, num_nodes=num_nodes, dropout_rate=0.5)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        N, C, T, V = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        x = self.layer1(x, self.A)
        x = self.layer2(x, self.A)
        x = self.layer3(x, self.A)
        x = self.layer4(x, self.A)
        x = self.layer5(x, self.A)
        x = self.layer6(x, self.A)

        x = x.mean(dim=-1).mean(dim=-1)
        return self.fc(x)
