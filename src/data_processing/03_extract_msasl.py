"""
Extract MediaPipe landmarks from MSASL frame-directory samples and merge into
wlasl_100_metadata.csv. Only processes the 53 glosses that overlap with our
WLASL-100 class set.

Each MSASL sample is a directory of JPEG frames (frame_000.jpg, frame_001.jpg…).
Output .npy files are saved as  landmarks/msasl_{gloss}_{sample_id}.npy
and new rows are appended to wlasl_100_metadata.csv.
"""
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm
import multiprocessing as MP

script_dir    = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "processed"))
MODELS_DIR    = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "models"))
MSASL_DIR     = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw", "msasl", "processdata"))

METADATA_CSV  = os.path.join(PROCESSED_DIR, "wlasl_100_metadata.csv")
LANDMARKS_DIR = os.path.join(PROCESSED_DIR, "landmarks")

POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

SPLIT_DIRS = {
    "train": "processed_data_MS_ASL100_Train",
    "val":   "processed_data_MS_ASL100_Val",
    "test":  "processed_data_MS_ASL100_Test",
}

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


def process_sample(args):
    """Process one MSASL sample directory. Returns 'ok' or 'fail'."""
    sample_dir, save_path = args

    frame_files = sorted(
        f for f in os.listdir(sample_dir) if f.endswith(".jpg")
    )
    frames = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(sample_dir, fname))
        if img is None:
            continue
        frames.append(extract_frame(img))

    if frames:
        tmp = save_path[:-4] + ".tmp.npy"  # np.save appends .npy if absent
        np.save(tmp, np.array(frames, dtype=np.float32))
        os.rename(tmp, save_path)
        return "ok"
    return "fail"


def build_task_list(overlap_glosses, label_map):
    """Return (tasks, new_rows) where tasks feed the pool and new_rows go into metadata."""
    tasks    = []
    new_rows = []

    for split, split_dirname in SPLIT_DIRS.items():
        split_dir = os.path.join(MSASL_DIR, split_dirname)
        if not os.path.isdir(split_dir):
            continue
        for gloss in overlap_glosses:
            gloss_dir = os.path.join(split_dir, gloss)
            if not os.path.isdir(gloss_dir):
                continue
            for sample_id in sorted(os.listdir(gloss_dir)):
                sample_dir = os.path.join(gloss_dir, sample_id)
                if not os.path.isdir(sample_dir):
                    continue
                video_id  = f"msasl_{gloss}_{sample_id}"
                save_path = os.path.join(LANDMARKS_DIR, f"{video_id}.npy")
                if os.path.exists(save_path):
                    continue
                tasks.append((sample_dir, save_path))
                new_rows.append({
                    "video_id":      video_id,
                    "gloss":         gloss,
                    "label_encoded": label_map[gloss],
                    "split":         split,
                    "fps":           -1,  # frames, not video fps
                })

    return tasks, new_rows


def main(num_workers):
    df        = pd.read_csv(METADATA_CSV)
    label_map = dict(zip(df["gloss"], df["label_encoded"]))
    existing_ids = set(df["video_id"].astype(str))

    msasl_glosses = set()
    for split_dirname in SPLIT_DIRS.values():
        d = os.path.join(MSASL_DIR, split_dirname)
        if os.path.isdir(d):
            msasl_glosses |= set(os.listdir(d))
    overlap = set(label_map.keys()) & msasl_glosses
    print(f"[*] {len(overlap)} overlapping glosses")

    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    tasks, new_rows = build_task_list(overlap, label_map)

    # Filter out rows already in metadata (re-run safety)
    new_rows = [r for r in new_rows if r["video_id"] not in existing_ids]

    print(f"[*] {len(tasks)} samples to extract")
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
            pool.imap_unordered(process_sample, tasks),
            total=len(tasks),
            desc="Extracting MSASL",
        ):
            if result == "ok":
                ok += 1
            else:
                fail += 1

    print(f"\n[*] Extracted: {ok} | Failed: {fail}")

    # Append new rows to metadata
    if new_rows:
        combined = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        combined.drop_duplicates("video_id", inplace=True)
        combined.to_csv(METADATA_CSV, index=False)
        print(f"[*] Metadata updated: {len(combined)} total rows (+{len(new_rows)} MSASL)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(num_workers=args.workers)
