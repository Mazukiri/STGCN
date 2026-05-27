import numpy as np

class LandmarkNormalizer:
    def __init__(self, target_frames=60):
        self.target_frames = target_frames

    def normalize_spatial(self, frames):
        """
        Chuẩn hóa không gian (Spatial Normalization).
        Input: numpy array shape (T, 75, 3)
        Output: numpy array shape (T, 75, 3) đã chuẩn hóa.
        Vectorized over T to avoid Python-level frame loop.
        """
        if len(frames) == 0:
            return frames

        frames = frames.copy()
        nose = frames[:, 0, :]              # (T, 3)
        valid = np.any(nose != 0, axis=1)  # (T,) — frames where pose was detected

        if not valid.any():
            return frames

        # Mask of detected landmarks per frame: (T, 75, 1) for broadcasting
        lm_mask = np.any(frames != 0, axis=2, keepdims=True)  # (T, 75, 1)

        # Center on nose for valid frames
        frames[valid] = np.where(
            lm_mask[valid],
            frames[valid] - nose[valid, np.newaxis, :],
            frames[valid],
        )

        # Scale by shoulder distance
        left_shoulder  = frames[:, 11, :]  # (T, 3) — already centered
        right_shoulder = frames[:, 12, :]  # (T, 3)
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=1)  # (T,)

        scale_valid = valid & (shoulder_dist > 1e-5)
        if scale_valid.any():
            scale = shoulder_dist[scale_valid, np.newaxis, np.newaxis]  # (T_v, 1, 1)
            frames[scale_valid] = np.where(
                lm_mask[scale_valid],
                frames[scale_valid] / scale,
                frames[scale_valid],
            )

        return frames

    def normalize_temporal(self, frames):
        """
        Chuẩn hóa thời gian (Temporal Normalization).
        Input: numpy array shape (T, 75, 3)
        Output: numpy array shape (target_frames, 75, 3)
        """
        T, num_landmarks, dims = frames.shape
        
        if T == self.target_frames:
            return frames
            
        if T < self.target_frames:
            # Zero-padding: chèn thêm tọa độ (0,0,0) vào cuối
            pad_len = self.target_frames - T
            padding = np.zeros((pad_len, num_landmarks, dims))
            return np.vstack((frames, padding))
            
        else:
            # Uniform Sampling (Lấy mẫu ngắt quãng dàn đều)
            indices = np.linspace(0, T - 1, self.target_frames, dtype=int)
            return frames[indices]

    def process(self, frames):
        """
        Kết hợp cả chuẩn hóa không gian và thời gian
        """
        frames = self.normalize_spatial(frames)
        frames = self.normalize_temporal(frames)
        return frames
