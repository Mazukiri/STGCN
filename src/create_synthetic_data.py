"""
Generates synthetic landmark data for verifying the training pipeline.
Run this when real WLASL data is unavailable.
Replace with the real pipeline (01_process_dataset.py + 02_extract_landmarks.py) when data is ready.
"""
import os
import json
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "processed"))
landmarks_dir = os.path.join(processed_dir, "landmarks")

os.makedirs(landmarks_dir, exist_ok=True)

NUM_CLASSES = 100
TOTAL_SAMPLES = 120  # 80 train / 20 val / 20 test
FRAMES = 45          # will be padded/sampled to 60 by normalizer
NODES = 75
DIMS = 3

label_names = [f"sign_{i:03d}" for i in range(NUM_CLASSES)]
label_mapping = {name: idx for idx, name in enumerate(label_names)}

splits = (
    ["train"] * 80 +
    ["val"] * 20 +
    ["test"] * 20
)

records = []
for i, split in enumerate(splits):
    video_id = f"synth_{i:04d}"
    label_idx = i % NUM_CLASSES
    gloss = label_names[label_idx]

    landmarks = np.random.randn(FRAMES, NODES, DIMS).astype(np.float32)
    # Ensure nose (index 0) is nonzero so spatial normalizer doesn't skip frames
    landmarks[:, 0, :] = np.random.uniform(0.1, 0.5, (FRAMES, DIMS))
    # Ensure shoulders (11, 12) are nonzero and separated so scale != 0
    landmarks[:, 11, :] = np.array([0.4, 0.5, 0.0])
    landmarks[:, 12, :] = np.array([0.6, 0.5, 0.0])

    np.save(os.path.join(landmarks_dir, f"{video_id}.npy"), landmarks)

    records.append({
        "video_id": video_id,
        "gloss": gloss,
        "label_encoded": label_idx,
        "split": split,
        "fps": 25,
    })

df = pd.DataFrame(records)
df.to_csv(os.path.join(processed_dir, "wlasl_100_metadata.csv"), index=False)

with open(os.path.join(processed_dir, "label_mapping.json"), "w") as f:
    json.dump(label_mapping, f, indent=4)

print(f"[*] Created {TOTAL_SAMPLES} synthetic samples in {landmarks_dir}")
print(f"    Train: {splits.count('train')}, Val: {splits.count('val')}, Test: {splits.count('test')}")
print(f"[*] Saved metadata and label_mapping.json to {processed_dir}")
