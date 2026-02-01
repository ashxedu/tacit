import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False, 
    model_complexity=1, 
    smooth_landmarks=True
)

def extract_keypoints(results):
    # Pose (33*4) + Left Hand (21*3) + Right Hand (21*3) = 258 landmarks
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
    
    # Get all folders in data/raw that contain videos
    labels = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    
    print(f"🚀 Found {len(labels)} classes: {labels}")
    
    total_videos = 0
    
    for label in labels:
        folder_path = os.path.join(RAW_DIR, label)
        # Filter for video extensions
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        
        print(f"   📂 Processing '{label}' ({len(video_files)} videos)...")
        
        for video_file in tqdm(video_files):
            input_path = os.path.join(folder_path, video_file)
            
            # Process video to extract skeleton
            sequence = process_video(input_path)
            
            # Save if valid data found
            if len(sequence) > 0:
                # Use splitext to handle filenames with multiple dots safely
                safe_filename = os.path.splitext(video_file)[0]
                save_name = f"{label}_{safe_filename}.npy"
                
                np.save(os.path.join(OUTPUT_DIR, save_name), np.array(sequence))
                total_videos += 1

    print(f"✅ DONE! Processed {total_videos} videos into {OUTPUT_DIR}")

if __name__ == "__main__":
    main()