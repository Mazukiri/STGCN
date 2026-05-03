import json
import os
import pandas as pd
from collections import Counter

# Set up paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw"))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "processed"))
JSON_PATH = os.path.join(RAW_DATA_DIR, "WLASL_v0.3.json")
OUTPUT_CSV = os.path.join(PROCESSED_DATA_DIR, "wlasl_100_metadata.csv")

def process_wlasl_json():
    print(f"[*] Reading JSON file from: {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        print(f"[!] Error: Could not find {JSON_PATH}")
        print("    Please download WLASL_v0.3.json and place it in the data/raw/ directory.")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[*] Total glosses in original JSON: {len(data)}")

    # Step 1: Count videos per gloss to find top 100
    gloss_counts = {}
    for entry in data:
        gloss = entry['gloss']
        num_videos = len(entry['instances'])
        gloss_counts[gloss] = num_videos

    # Get top 100 glosses
    top_100_glosses = [g for g, count in Counter(gloss_counts).most_common(100)]
    print(f"[*] Selected top 100 glosses. Min videos for a gloss in top 100: {gloss_counts[top_100_glosses[-1]]}")

    # Step 2: Extract metadata for top 100
    records = []
    label_mapping = {gloss: idx for idx, gloss in enumerate(top_100_glosses)}

    for entry in data:
        gloss = entry['gloss']
        if gloss in top_100_glosses:
            label_encoded = label_mapping[gloss]
            for instance in entry['instances']:
                video_id = instance['video_id']
                split = instance['split'] # 'train', 'val', or 'test'
                fps = instance.get('fps', 25)
                
                records.append({
                    'video_id': video_id,
                    'gloss': gloss,
                    'label_encoded': label_encoded,
                    'split': split,
                    'fps': fps
                })

    df = pd.DataFrame(records)
    print(f"[*] Extracted {len(df)} videos for WLASL-100.")
    print(f"    Train: {len(df[df['split']=='train'])}, Val: {len(df[df['split']=='val'])}, Test: {len(df[df['split']=='test'])}")

    # Step 3: Save to CSV
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[*] Saved metadata to: {OUTPUT_CSV}")

    # Save Label Mapping
    mapping_path = os.path.join(PROCESSED_DATA_DIR, "label_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=4)
    print(f"[*] Saved label mapping to: {mapping_path}")

if __name__ == "__main__":
    process_wlasl_json()
