import json
import os
import cv2
import mediapipe as mp
import numpy as np

# --- CONFIGURATION & PATH SETUP ---
# Get the absolute path of the root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to the root
RAW_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "videos")
JSON_PATH = os.path.join(BASE_DIR, "data", "raw", "WLASL_v0.3.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# The "Golden 10" target words and their aliases in WLASL
TARGET_WORDS = {
    "hello": ["hello"],
    "goodbye": ["goodbye", "bye"],
    "please": ["please"],
    "thank you": ["thank you", "thanks"],
    "sorry": ["sorry"],
    "yes": ["yes"],
    "no": ["no"],
    "help": ["help"],
    "go": ["go"],
    "stop": ["stop"]
}

# Parameters
MIN_VIDEOS = 15      # Minimum samples required for validity
MAX_VIDEOS = 60      # Cap to prevent class imbalance

# Setup MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False, 
    model_complexity=1, 
    smooth_landmarks=True
)

def extract_keypoints(results):
    """ 
    Extracts normalized coordinates (x, y, z) from MediaPipe results.
    Returns a flattened concatenated NumPy array.
    """
    # 1. Face
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

def process_video(video_path):
    """ Reads video, runs MediaPipe, returns sequence of frames """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get total frames to calculate progress if needed
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Optimization: Resize to 256x256 if not already
        # MediaPipe needs RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = holistic.process(image)
        
        # Extraction
        keypoints = extract_keypoints(results)
        frames.append(keypoints)
        
    cap.release()
    return frames

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"📂 Loading WLASL Index from: {JSON_PATH}")
    
    try:
        with open(JSON_PATH, 'r') as f:
            wlasl_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Could not find JSON at {JSON_PATH}")
        return

    stats = {}
    
    for entry in wlasl_data:
        gloss = entry['gloss']
        
        # Map dataset gloss to our target label
        target_label = None
        for label, aliases in TARGET_WORDS.items():
            if gloss in aliases:
                target_label = label
                break
        
        if not target_label:
            continue

        if target_label not in stats:
            stats[target_label] = 0

        # Cap data to prevent class imbalance
        if stats[target_label] >= MAX_VIDEOS:
            continue

        for instance in entry['instances']:
            video_id = instance['video_id']
            # Search for the video file
            video_path = os.path.join(RAW_VIDEO_DIR, f"{video_id}.mp4")

            if os.path.exists(video_path):
                print(f"   🎥 Processing {target_label.upper()} ({video_id})...")
                
                sequence = process_video(video_path)
                
                # Validity Check: Must have frames
                if len(sequence) > 0:
                    save_name = f"{target_label}_{video_id}.npy"
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    np.save(save_path, np.array(sequence))
                    
                    stats[target_label] += 1
                else:
                    print("   ⚠️ Empty/Corrupt video, skipping.")
            
            if stats[target_label] >= MAX_VIDEOS: 
                break

    # Final Report
    print("\n" + "="*40)
    print("📊 DATASET EXTRACTION REPORT")
    print("="*40)
    
    valid_words = []
    for word, count in stats.items():
        status = "✅ READY" if count >= MIN_VIDEOS else "❌ INSUFFICIENT"
        print(f"{word.upper():<15} : {count} samples \t {status}")
        if count >= MIN_VIDEOS:
            valid_words.append(word)

    print("-" * 40)
    print(f"Total Classes Ready: {len(valid_words)} / 10")
    print("="*40)

if __name__ == "__main__":
    main()