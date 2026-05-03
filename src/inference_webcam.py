import cv2
import mediapipe as mp
import torch
import json
import numpy as np
import os
from collections import deque
from models.stgcn import STGCN
from data_processing.normalization import LandmarkNormalizer

def load_label_mapping(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    # Reverse mapping: index -> label string
    return {v: k for k, v in mapping.items()}

def extract_landmarks(results):
    """Trích xuất 75 điểm (33 Pose + 21 Left Hand + 21 Right Hand)"""
    pose = np.zeros((33, 3))
    lh = np.zeros((21, 3))
    rh = np.zeros((21, 3))
    
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i] = [lm.x, lm.y, lm.z]
            
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i] = [lm.x, lm.y, lm.z]
            
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i] = [lm.x, lm.y, lm.z]
            
    return np.concatenate([pose, lh, rh])

def main():
    # 1. Khởi tạo đường dẫn
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "weights", "best_stgcn_model.pth")
    mapping_path = os.path.join(script_dir, "..", "data", "processed", "label_mapping.json")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Đang chạy Real-time Inference trên: {device}")
    
    # 2. Tải nhãn từ vựng
    if not os.path.exists(mapping_path):
        print(f"[!] Lỗi: Không tìm thấy file {mapping_path}.")
        print("[!] Vui lòng chạy kịch bản 01_process_dataset.py ở Giai đoạn 1 trước.")
        return
    idx_to_label = load_label_mapping(mapping_path)
    num_classes = len(idx_to_label)
    
    # 3. Khởi tạo mạng nơ-ron ST-GCN
    model = STGCN(in_channels=3, num_classes=num_classes)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("[*] Đã tải thành công trọng số mạng ST-GCN.")
    else:
        print("[!] Không tìm thấy trọng số best_stgcn_model.pth. Cần train mô hình trước.")
    
    model.to(device)
    model.eval()
    
    # 4. Cơ chế Cửa sổ trượt (Sliding Window)
    TARGET_FRAMES = 60
    frames_queue = deque(maxlen=TARGET_FRAMES)
    normalizer = LandmarkNormalizer(target_frames=TARGET_FRAMES)
    
    current_prediction = "Waiting..."
    current_confidence = 0.0
    
    # 5. Mở OpenCV và MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0) # 0 là Camera mặc định của máy tính
    
    if not cap.isOpened():
        print("[!] Lỗi: Không thể kết nối với Webcam. Vui lòng kiểm tra lại thiết bị.")
        return
        
    print("[*] Đã mở Webcam. Hãy giơ tay lên để biểu diễn ngôn ngữ ký hiệu. Bấm 'q' để thoát.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # OpenCV đọc ảnh BGR, MediaPipe cần RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Trích xuất tọa độ
            landmarks = extract_landmarks(results)
            
            # Chỉ ghi nhận vào bộ đệm nếu nhận diện được cơ thể người (chống nhiễu)
            if np.sum(landmarks[:33]) != 0:
                frames_queue.append(landmarks)
            else:
                if len(frames_queue) > 0:
                    frames_queue.clear()
            
            # 6. Kích hoạt mạng Nơ-ron khi đã thu đủ 60 frames
            if len(frames_queue) == TARGET_FRAMES:
                frames_np = np.array(frames_queue)
                
                # Chuẩn hóa (tịnh tiến gốc tọa độ, scale theo vai)
                norm_frames = normalizer.process(frames_np)
                
                # Transform shape cho ST-GCN: (T, 75, 3) -> (Batch=1, Channels=3, Frames=60, Nodes=75)
                input_tensor = np.transpose(norm_frames, (2, 0, 1))
                input_tensor = np.expand_dims(input_tensor, axis=0)
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(device)
                
                # Đưa vào ST-GCN dự đoán
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    
                    current_confidence = confidence.item()
                    if current_confidence > 0.6: # Ngưỡng tự tin 60%
                        current_prediction = idx_to_label[predicted_class.item()]
                    else:
                        current_prediction = "..."
                        
            # 7. Render Đồ họa
            # Vẽ các đường xương
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Vẽ thanh trạng thái
            cv2.rectangle(image_bgr, (0, 0), (640, 60), (0, 0, 0), -1)
            
            # In nhãn
            text = f'AI: {current_prediction.upper()} ({current_confidence*100:.1f}%)'
            cv2.putText(image_bgr, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # In trạng thái bộ đệm
            cv2.putText(image_bgr, f'Buffer: {len(frames_queue)}/60', (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow('Sign Language AI Demo', image_bgr)
            
            # Thoát bằng phím q
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
