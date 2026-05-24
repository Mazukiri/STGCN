import os
import argparse
import subprocess
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm
import multiprocessing as MP

script_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR   = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw"))
PROCESSED_DIR  = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "processed"))
MODELS_DIR     = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "models"))

METADATA_CSV  = os.path.join(PROCESSED_DIR, "wlasl_100_metadata.csv")
VIDEO_DIR     = os.path.join(RAW_DATA_DIR,  "videos")
LANDMARKS_DIR = os.path.join(PROCESSED_DIR, "landmarks")

POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for url, path in [(POSE_MODEL_URL, POSE_MODEL_PATH), (HAND_MODEL_URL, HAND_MODEL_PATH)]:
        if not os.path.exists(path):
            print(f"[*] Downloading {os.path.basename(path)}...")
            subprocess.run(["curl", "-L", "-o", path, url], check=True)


# Per-process detector instances (created once in each worker via worker_init)
_pose_det = None
_hand_det = None


def worker_init(pose_path, hand_path):
    global _pose_det, _hand_det
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    _pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)
    _hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)


def extract_frame(image_bgr):
    """Extract (75, 3) landmarks from one BGR frame."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    pose = np.zeros((33, 3))
    lh   = np.zeros((21, 3))
    rh   = np.zeros((21, 3))

    pr = _pose_det.detect(mp_img)
    if pr.pose_landmarks:
        for i, lm in enumerate(pr.pose_landmarks[0]):
            pose[i] = [lm.x, lm.y, lm.z]

    hr = _hand_det.detect(mp_img)
    if hr.hand_landmarks:
        for landmarks, handedness in zip(hr.hand_landmarks, hr.handedness):
            arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            if handedness[0].category_name == "Left":
                lh = arr
            else:
                rh = arr

    return np.concatenate([pose, lh, rh])  # (75, 3)


def process_video(args):
    """Process one .mp4 video. Returns 'ok' or 'fail'."""
    video_id, video_path, save_path = args

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extract_frame(frame))
    cap.release()

    if frames:
        tmp = save_path[:-4] + ".tmp.npy"  # np.save appends .npy if absent
        np.save(tmp, np.array(frames, dtype=np.float32))
        os.rename(tmp, save_path)
        return "ok"
    return "fail"


def process_videos(num_workers):
    if not os.path.exists(METADATA_CSV):
        print(f"[!] Error: {METADATA_CSV} not found.")
        return

    download_models()

    df = pd.read_csv(METADATA_CSV)
    os.makedirs(LANDMARKS_DIR, exist_ok=True)

    # Build task list — skip already-extracted videos
    tasks = []
    for _, row in df.iterrows():
        video_id  = row["video_id"]
        save_path = os.path.join(LANDMARKS_DIR, f"{video_id}.npy")
        if os.path.exists(save_path):
            continue
        video_path = os.path.join(VIDEO_DIR, f"{str(video_id).zfill(5)}.mp4")
        if not os.path.exists(video_path):
            continue
        tasks.append((video_id, video_path, save_path))

    already_done = len(df) - len(tasks)
    print(f"[*] {len(tasks)} to extract | {already_done} already done")

    if not tasks:
        print("[*] Nothing to do.")
        return

    ctx = MP.get_context("spawn")
    ok = fail = 0

    with ctx.Pool(
        processes=num_workers,
        initializer=worker_init,
        initargs=(POSE_MODEL_PATH, HAND_MODEL_PATH),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(process_video, tasks),
            total=len(tasks),
            desc="Extracting",
        ):
            if result == "ok":
                ok += 1
            else:
                fail += 1

    print(f"\n[*] Done. Extracted: {ok} | Failed: {fail}")
    print(f"[*] Landmarks saved to: {LANDMARKS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    args = parser.parse_args()
    process_videos(num_workers=args.workers)
