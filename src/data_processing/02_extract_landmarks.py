import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw"))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "processed"))

METADATA_CSV = os.path.join(PROCESSED_DATA_DIR, "wlasl_100_metadata.csv")
VIDEO_DIR = os.path.join(RAW_DATA_DIR, "videos")
LANDMARKS_DIR = os.path.join(PROCESSED_DATA_DIR, "landmarks")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def extract_landmarks(image, holistic):
    """
    Trích xuất tọa độ Pose, Left Hand, Right Hand từ 1 ảnh (khung hình).
    Trả về mảng NumPy shape (75, 3) tương ứng 75 điểm (x, y, z).
    Nếu điểm nào không thấy, gán (0, 0, 0).
    """
    # Convert BGR (OpenCV) to RGB (MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    
    # Initialize empty arrays
    pose = np.zeros((33, 3))
    lh = np.zeros((21, 3))
    rh = np.zeros((21, 3))
    
    # Pose Landmarks (33 points)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i] = [lm.x, lm.y, lm.z]
            
    # Left Hand Landmarks (21 points)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i] = [lm.x, lm.y, lm.z]
            
    # Right Hand Landmarks (21 points)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i] = [lm.x, lm.y, lm.z]
            
    # Concatenate all into a single frame representation
    # Shape: (33 + 21 + 21, 3) = (75, 3)
    frame_landmarks = np.concatenate([pose, lh, rh])
    return frame_landmarks

def process_videos():
    if not os.path.exists(METADATA_CSV):
        print(f"[!] Error: Could not find {METADATA_CSV}")
        print("    Please run 01_process_dataset.py first.")
        return
        
    df = pd.read_csv(METADATA_CSV)
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    
    successful_videos = []
    missing_videos = []
    
    print("[*] Starting landmark extraction using MediaPipe Holistic...")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            video_id = row['video_id']
            video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
            save_path = os.path.join(LANDMARKS_DIR, f"{video_id}.npy")
            
            # Bỏ qua nếu đã xử lý từ trước
            if os.path.exists(save_path):
                successful_videos.append(video_id)
                continue
                
            if not os.path.exists(video_path):
                missing_videos.append(video_id)
                continue
                
            cap = cv2.VideoCapture(video_path)
            video_landmarks = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_landmarks = extract_landmarks(frame, holistic)
                video_landmarks.append(frame_landmarks)
                
            cap.release()
            
            # Chỉ lưu file .npy nếu video có khung hình
            if len(video_landmarks) > 0:
                video_landmarks_np = np.array(video_landmarks)
                np.save(save_path, video_landmarks_np)
                successful_videos.append(video_id)
                
    print(f"[*] Extraction complete.")
    print(f"    Successfully processed: {len(successful_videos)}/{len(df)} videos.")
    if len(missing_videos) > 0:
        print(f"    [!] Missing {len(missing_videos)} videos in {VIDEO_DIR}.")
    print(f"[*] Landmarks saved to: {LANDMARKS_DIR}")

if __name__ == "__main__":
    process_videos()
