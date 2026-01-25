import os
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "ASL_Citizen")
VIDEOS_DIR = os.path.join(DATA_RAW_DIR, "videos")
SPLITS_DIR = os.path.join(DATA_RAW_DIR, "splits")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# TARGETS: Map "Clean Label" -> "Search Term"
SEARCH_TERMS = {
    "hello": "hello",
    "goodbye": "bye",       
    "please": "please",
    "intelligent": "intelligent", 
    "sorry": "sorry",
    "yes": "yes",
    "no": "no",
    "help": "help",
    "go": "go",
    "stop": "stop"
}

MAX_SAMPLES = 50 

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        frames.append(extract_keypoints(results))
    cap.release()
    return frames

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # 1. Load CSVs
    dfs = []
    for split in ['train.csv', 'val.csv', 'test.csv']:
        path = os.path.join(SPLITS_DIR, split)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    
    if not dfs:
        print("❌ Error: No CSV files found.")
        return

    full_df = pd.concat(dfs)
    full_df.columns = [c.lower() for c in full_df.columns]
    
    if 'gloss' not in full_df.columns:
        print(f"❌ Error: 'gloss' column missing. Found: {full_df.columns}")
        return

    full_df['clean_gloss'] = full_df['gloss'].astype(str).str.lower().str.strip()
    stats = {word: 0 for word in SEARCH_TERMS.keys()}

    print(f"🚀 Scanning ASL Citizen Index ({len(full_df)} rows)...")

    for target_label, search_term in SEARCH_TERMS.items():
        subset = full_df[full_df['clean_gloss'].str.contains(search_term, na=False)]
        
        # Strict check for "no" to avoid "know" or "now"
        if target_label == "no": 
            subset = full_df[full_df['clean_gloss'] == "no"]

        print(f"   🔍 Processing '{target_label.upper()}' (Found {len(subset)} candidates)...")
        
        for _, row in subset.iterrows():
            if stats[target_label] >= MAX_SAMPLES: break
            
            video_filename = row['video file']
            video_path = os.path.join(VIDEOS_DIR, video_filename)

            if os.path.exists(video_path):
                seq = process_video(video_path)
                if len(seq) > 0:
                    save_name = f"{target_label}_{video_filename.replace('.mp4', '')}.npy"
                    np.save(os.path.join(OUTPUT_DIR, save_name), np.array(seq))
                    stats[target_label] += 1
            
    print("\n📊 EXTRACTION REPORT:")
    for word, count in stats.items():
        status = "✅ READY" if count >= 15 else "❌ LOW DATA"
        print(f"{word.upper():<15} : {count} samples \t {status}")

if __name__ == "__main__":
    main()