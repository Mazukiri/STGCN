import numpy as np

class LandmarkNormalizer:
    def __init__(self, target_frames=60):
        self.target_frames = target_frames

    def normalize_spatial(self, frames):
        """
        Chuẩn hóa không gian (Spatial Normalization).
        Input: numpy array shape (T, 75, 3) (T = frames, 75 = landmarks, 3 = x,y,z)
        Output: numpy array shape (T, 75, 3) đã chuẩn hóa.
        """
        if len(frames) == 0:
            return frames
            
        normalized_frames = np.zeros_like(frames)
        
        for t in range(frames.shape[0]):
            frame_data = frames[t].copy()
            
            # MediaPipe Pose: Mũi (Nose) là index 0
            nose = frame_data[0]
            
            # MediaPipe Pose: Vai phải (Right Shoulder) là index 12, Vai trái (Left Shoulder) là index 11
            left_shoulder = frame_data[11]
            right_shoulder = frame_data[12]
            
            # Nếu khung hình này không nhận diện được pose (tọa độ 0)
            if np.all(nose == 0):
                normalized_frames[t] = frame_data
                continue
                
            # Tịnh tiến gốc tọa độ về Mũi
            # Chỉ dịch chuyển các điểm không phải là (0,0,0) - tức là các điểm nhận diện được
            mask = np.any(frame_data != 0, axis=1)
            frame_data[mask] = frame_data[mask] - nose
            
            # Tính khoảng cách giữa 2 vai để scale
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
            
            # Tránh chia cho 0
            if shoulder_dist > 1e-5:
                frame_data[mask] = frame_data[mask] / shoulder_dist
                
            normalized_frames[t] = frame_data
            
        return normalized_frames

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
