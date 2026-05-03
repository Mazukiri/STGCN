import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(self, input_size=225, hidden_size=256, num_layers=2, num_classes=100, dropout=0.5):
        """
        Khởi tạo mạng LSTM cơ bản.
        - input_size: 75 landmarks * 3 dims = 225
        - num_classes: Mặc định 100 lớp cho tập WLASL-100
        """
        super(BaselineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True => Input tensor có dạng (Batch, Seq_Len, Input_Size)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Lớp Dropout chống Overfitting (học vẹt)
        self.dropout = nn.Dropout(dropout)
        
        # Lớp mạng nơ-ron dày đặc (Fully Connected) để phân loại
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Đầu vào x có kích thước: (Batch, Seq_Len, Input_Size)
        
        # Truyền qua mạng LSTM
        # out có kích thước: (Batch, Seq_Len, Hidden_Size)
        out, _ = self.lstm(x)
        
        # Chỉ lấy kết quả của LSTM ở bước thời gian cuối cùng (last timestep)
        # Vì ta mong muốn nhận diện cả 1 từ sau khi xem hết video
        out = out[:, -1, :] 
        
        # Đưa qua Dropout và FC layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
