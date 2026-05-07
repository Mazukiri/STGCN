import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data_processing.normalization import LandmarkNormalizer

class WLASLDataset(Dataset):
    def __init__(self, metadata_path, landmarks_dir, split='train', target_frames=60, return_graph=False):
        """
        Khởi tạo Dataset.
        - return_graph: True nếu xuất ra cho ST-GCN (Channels, Frames, Nodes)
                      False nếu xuất ra cho LSTM (Frames, Channels*Nodes)
        """
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        self.landmarks_dir = landmarks_dir
        self.normalizer = LandmarkNormalizer(target_frames=target_frames)
        self.return_graph = return_graph
        
        valid_indices = []
        for idx in range(len(self.metadata)):
            video_id = self.metadata.iloc[idx]['video_id']
            file_path = os.path.join(self.landmarks_dir, f"{video_id}.npy")
            if os.path.exists(file_path):
                valid_indices.append(idx)
        
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        video_id = row['video_id']
        label = row['label_encoded']
        
        file_path = os.path.join(self.landmarks_dir, f"{video_id}.npy")
        frames = np.load(file_path)
        
        # Áp dụng chuẩn hóa không gian và thời gian -> Shape: (T, 75, 3)
        frames_normalized = self.normalizer.process(frames)
        T, num_landmarks, dims = frames_normalized.shape
        
        if self.return_graph:
            # Cho ST-GCN: Cần cấu trúc (Channels, Frames, Nodes) -> (3, T, 75)
            # Permute từ (T, 75, 3) thành (3, T, 75)
            frames_out = np.transpose(frames_normalized, (2, 0, 1))
        else:
            # Cho LSTM: Flatten (T, 225)
            frames_out = frames_normalized.reshape(T, num_landmarks * dims)
        
        x = torch.tensor(frames_out, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y

def get_dataloaders(metadata_path, landmarks_dir, batch_size=32, target_frames=60, return_graph=False):
    train_dataset = WLASLDataset(metadata_path, landmarks_dir, split='train', target_frames=target_frames, return_graph=return_graph)
    val_dataset = WLASLDataset(metadata_path, landmarks_dir, split='val', target_frames=target_frames, return_graph=return_graph)
    test_dataset = WLASLDataset(metadata_path, landmarks_dir, split='test', target_frames=target_frames, return_graph=return_graph)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
