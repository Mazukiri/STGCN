import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data_processing.normalization import LandmarkNormalizer

# MediaPipe landmark layout: 33 pose (0-32), 21 left hand (33-53), 21 right hand (54-74)
_LEFT_HAND  = slice(33, 54)
_RIGHT_HAND = slice(54, 75)


class WLASLDataset(Dataset):
    def __init__(self, metadata_path, landmarks_dir, split='train', target_frames=60,
                 return_graph=False, augment=False):
        """
        - return_graph: True → ST-GCN (Channels, Frames, Nodes)
                        False → LSTM (Frames, Channels*Nodes)
        - augment: apply random transforms per __getitem__ call (train only)
        Raw normalized frames stored as numpy; conversion happens in __getitem__.
        """
        metadata = pd.read_csv(metadata_path)
        metadata = metadata[metadata['split'] == split].reset_index(drop=True)
        normalizer = LandmarkNormalizer(target_frames=target_frames)

        self.target_frames = target_frames
        self.return_graph = return_graph
        self.augment = augment
        self._normalizer = normalizer
        self.samples = []

        for _, row in metadata.iterrows():
            file_path = os.path.join(landmarks_dir, f"{row['video_id']}.npy")
            if not os.path.exists(file_path):
                continue
            frames = normalizer.process(np.load(file_path))  # (60, 75, 3) float32
            label  = torch.tensor(int(row['label_encoded']), dtype=torch.long)
            self.samples.append((frames, label))             # store numpy for augment

    def _augment(self, frames):
        frames = frames.copy()

        # 1. Horizontal mirror (p=0.5): flip x, swap left↔right hand nodes
        if np.random.random() < 0.5:
            frames[:, :, 0] *= -1
            tmp = frames[:, _LEFT_HAND, :].copy()
            frames[:, _LEFT_HAND, :]  = frames[:, _RIGHT_HAND, :]
            frames[:, _RIGHT_HAND, :] = tmp

        # 2. Gaussian noise on detected landmarks only (p=0.5, σ=0.01)
        if np.random.random() < 0.5:
            mask = np.any(frames != 0, axis=2, keepdims=True)  # (60, 75, 1)
            frames += np.where(mask, np.random.normal(0, 0.01, frames.shape), 0).astype(np.float32)

        # 3. Time stretch (p=0.3, factor ∈ [0.8, 1.2]) then re-pad/resample to target_frames
        if np.random.random() < 0.3:
            T = frames.shape[0]
            T_new = int(T * np.random.uniform(0.8, 1.2))
            if T_new > 1:
                idx = np.linspace(0, T - 1, T_new, dtype=int)
                frames = frames[idx]
                frames = self._normalizer.normalize_temporal(frames)

        # 4. Joint masking (p=0.3): zero out 10% of nodes to simulate missing detections
        if np.random.random() < 0.3:
            n_mask = max(1, int(0.1 * frames.shape[1]))
            mask_idx = np.random.choice(frames.shape[1], n_mask, replace=False)
            frames[:, mask_idx, :] = 0.0

        # 5. Temporal shift (p=0.3): shift ±10 frames with edge-padding
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 11)
            frames = np.roll(frames, shift, axis=0)
            if shift > 0:
                frames[:shift] = frames[shift]
            elif shift < 0:
                frames[shift:] = frames[shift - 1]

        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        if self.augment:
            frames = self._augment(frames)
        if self.return_graph:
            frames = np.transpose(frames, (2, 0, 1))           # (3, 60, 75)
        else:
            frames = frames.reshape(self.target_frames, -1)    # (60, 225)
        return torch.tensor(frames, dtype=torch.float32), label


def get_dataloaders(metadata_path, landmarks_dir, batch_size=32, target_frames=60,
                    return_graph=False, num_workers=4, pin_memory=False):
    kwargs = dict(metadata_path=metadata_path, landmarks_dir=landmarks_dir,
                  target_frames=target_frames, return_graph=return_graph)
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = DataLoader(WLASLDataset(split='train', augment=True,  **kwargs),
                              batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(WLASLDataset(split='val',   augment=False, **kwargs),
                              batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(WLASLDataset(split='test',  augment=False, **kwargs),
                              batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
