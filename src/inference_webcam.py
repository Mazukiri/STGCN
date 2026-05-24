import cv2
import mediapipe as mp
import torch
import json
import numpy as np
import os
import time
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from models.stgcn import STGCN
from data_processing.normalization import LandmarkNormalizer


def load_label_mapping(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return {v: k for k, v in mapping.items()}


def extract_landmarks(image_bgr, pose_det, hand_det, timestamp_ms):
    """Extract (75, 3) landmarks — matches 02_extract_landmarks.py pipeline exactly.
    Returns (landmarks_75x3, pose_result, hand_result)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    pose = np.zeros((33, 3))
    lh   = np.zeros((21, 3))
    rh   = np.zeros((21, 3))

    pr = pose_det.detect_for_video(mp_img, timestamp_ms)
    if pr.pose_landmarks:
        for i, lm in enumerate(pr.pose_landmarks[0]):
            pose[i] = [lm.x, lm.y, lm.z]

    hr = hand_det.detect_for_video(mp_img, timestamp_ms)
    if hr.hand_landmarks:
        for landmarks, handedness in zip(hr.hand_landmarks, hr.handedness):
            arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            if handedness[0].category_name == "Left":
                lh = arr
            else:
                rh = arr

    return np.concatenate([pose, lh, rh]), pr, hr


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "weights", "best_stgcn_model.pth")
    mapping_path = os.path.join(script_dir, "..", "data", "processed", "label_mapping.json")
    models_dir   = os.path.join(script_dir, "..", "data", "models")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[*] Device: {device}")

    if not os.path.exists(mapping_path):
        print(f"[!] Label mapping not found: {mapping_path}")
        return

    idx_to_label = load_label_mapping(mapping_path)
    num_classes  = len(idx_to_label)

    model = STGCN(in_channels=3, num_classes=num_classes)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("[*] Weights loaded.")
    else:
        print(f"[!] Weights not found: {weights_path}")
        return
    model.to(device)
    model.eval()

    pose_model_path = os.path.join(models_dir, "pose_landmarker_lite.task")
    hand_model_path = os.path.join(models_dir, "hand_landmarker.task")
    for path in [pose_model_path, hand_model_path]:
        if not os.path.exists(path):
            print(f"[!] Model file missing: {path}")
            print("[!] Run src/data_processing/02_extract_landmarks.py first.")
            return

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)
    hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)
    print("[*] MediaPipe detectors ready.")

    TARGET_FRAMES = 60
    RESET_THRESHOLD = 15
    frames_queue = deque(maxlen=TARGET_FRAMES)
    normalizer = LandmarkNormalizer(target_frames=TARGET_FRAMES)

    current_prediction = "Waiting..."
    current_confidence = 0.0
    no_detection_count = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Cannot open webcam.")
        pose_det.close()
        hand_det.close()
        return

    print("[*] Webcam open. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(time.time() * 1000)
        landmarks, pr, hr = extract_landmarks(frame, pose_det, hand_det, timestamp_ms)

        # Draw pose skeleton
        if pr.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pr.pose_landmarks[0],
                PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Draw hand skeletons
        for hand_lms in hr.hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_lms,
                HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
            )

        if np.sum(landmarks[:33]) == 0:
            no_detection_count += 1
            if no_detection_count >= RESET_THRESHOLD:
                frames_queue.clear()
                no_detection_count = 0
        else:
            no_detection_count = 0
            frames_queue.append(landmarks)

        if len(frames_queue) == TARGET_FRAMES:
            frames_np = np.array(frames_queue)
            norm_frames = normalizer.process(frames_np)

            input_tensor = torch.tensor(
                np.expand_dims(np.transpose(norm_frames, (2, 0, 1)), axis=0),
                dtype=torch.float32,
            ).to(device)

            with torch.no_grad():
                probs = torch.softmax(model(input_tensor), dim=1)
                confidence, predicted_class = torch.max(probs, 1)
                current_confidence = confidence.item()
                if current_confidence > 0.6:
                    current_prediction = idx_to_label[predicted_class.item()]
                else:
                    current_prediction = "..."

        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        text = f'AI: {current_prediction.upper()} ({current_confidence * 100:.1f}%)'
        cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Buffer: {len(frames_queue)}/60', (500, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Sign Language AI Demo', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    pose_det.close()
    hand_det.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
