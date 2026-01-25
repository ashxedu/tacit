import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODELS_PATH = os.path.join(BASE_DIR, "training", "models")
LOG_DIR = os.path.join(BASE_DIR, "training", "logs")

# Increased threshold (more data now)
MIN_SAMPLES_REQUIRED = 20 

# Model Params
# reduced epochs slightly since we have more/better data
EPOCHS = 120  
BATCH_SIZE = 32 # Increased batch size for stable gradient descent

def load_data():
    """ 
    Loads .npy files, filters out classes, and returns (X, y, categories).
    """
    sequences, labels = [], []
    file_list = os.listdir(DATA_PATH)
    
    # 1. Count samples per class
    class_counts = {}
    for filename in file_list:
        if not filename.endswith(".npy"): continue
        word = filename.split('_')[0]
        class_counts[word] = class_counts.get(word, 0) + 1
        
    # 2. Filter valid classes AND SORT THEM (CRITICAL FIX)
    # Sorting ensures 0=Go, 1=Goodbye, etc. consistently
    valid_classes = sorted([w for w, c in class_counts.items() if c >= MIN_SAMPLES_REQUIRED])
    
    # Create the map: {'go': 0, 'goodbye': 1, ...}
    label_map = {label: num for num, label in enumerate(valid_classes)}
    
    print(f"✅ Training on {len(valid_classes)} classes (Alphabetical):")
    print(valid_classes)
    
    # 3. Load and Normalize
    for filename in file_list:
        word = filename.split('_')[0]
        if word not in valid_classes: continue
        
        path = os.path.join(DATA_PATH, filename)
        res = np.load(path)
        
        # Standardize length to 30 frames
        target_length = 30
        if len(res) > target_length:
            res = res[:target_length]
        elif len(res) < target_length:
            padding = np.zeros((target_length - len(res), res.shape[1]))
            res = np.concatenate((res, padding))
            
        sequences.append(res)
        labels.append(label_map[word])
        
    return np.array(sequences), np.array(labels), valid_classes

def build_model(input_shape, num_classes):
    """ Builds the LSTM Neural Network """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    if not os.path.exists(MODELS_PATH): os.makedirs(MODELS_PATH)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    # 1. Load Data
    X, y, classes = load_data()
    
    if len(classes) == 0:
        print("❌ Error: No valid classes found. Check DATA_PATH.")
        return

    # 2. One-Hot Encoding
    y = to_categorical(y).astype(int)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    print(f"🧠 Data Shape: {X.shape}")
    
    # 4. Build & Train
    model = build_model(X.shape[1:], len(classes))
    tb_callback = TensorBoard(log_dir=LOG_DIR)
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tb_callback], validation_data=(X_test, y_test))
    
    # 5. Save Model
    print("💾 Saving Model...")
    model.save(os.path.join(MODELS_PATH, 'tacit_brain.h5'))
    
    # Save the class labels 
    import pickle
    with open(os.path.join(MODELS_PATH, 'classes.pkl'), 'wb') as f:
        pickle.dump(classes, f)
        
    print("✅ Training Complete.")
    print(f"📋 FINAL CLASS ORDER (Copy this to React): {classes}")

if __name__ == "__main__":
    main()