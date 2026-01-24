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

# We lower the threshold to 10 to include 'Stop' (11 samples) and 'Hello' (13)
MIN_SAMPLES_REQUIRED = 10 

# Model Param
EPOCHS = 150  # High epochs because we have small data
BATCH_SIZE = 16

def load_data():
    """ 
    Loads .npy files, filters out classes with < 10 samples, 
    and returns (X, y, categories).
    """
    sequences, labels = [], []
    file_list = os.listdir(DATA_PATH)
    
    # 1. Count samples per class
    class_counts = {}
    for filename in file_list:
        if not filename.endswith(".npy"): continue
        word = filename.split('_')[0]
        class_counts[word] = class_counts.get(word, 0) + 1
        
    # 2. Filter valid classes
    valid_classes = [w for w, c in class_counts.items() if c >= MIN_SAMPLES_REQUIRED]
    label_map = {label: num for num, label in enumerate(valid_classes)}
    
    print(f"✅ Training on {len(valid_classes)} classes: {valid_classes}")
    
    # 3. Load and Normalize
    for filename in file_list:
        word = filename.split('_')[0]
        if word not in valid_classes: continue
        
        path = os.path.join(DATA_PATH, filename)
        res = np.load(path)
        
        # Standardize length to 30 frames (pad or cut)
        target_length = 30
        if len(res) > target_length:
            res = res[:target_length]
        elif len(res) < target_length:
            # Zero padding
            padding = np.zeros((target_length - len(res), res.shape[1]))
            res = np.concatenate((res, padding))
            
        sequences.append(res)
        labels.append(label_map[word])
        
    return np.array(sequences), np.array(labels), valid_classes

def build_model(input_shape, num_classes):
    """ Builds the LSTM Neural Network """
    model = Sequential()
    
    # Layer 1: LSTM looking for patterns over time
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    
    # Layer 2: LSTM distilling the patterns
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    
    # Layer 3: Dense layers for decision making
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    
    # Output Layer: One neuron per word
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    if not os.path.exists(MODELS_PATH): os.makedirs(MODELS_PATH)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    # 1. Load Data
    X, y, classes = load_data()
    
    # 2. Convert labels to binary matrix (One-Hot Encoding)
    y = to_categorical(y).astype(int)
    
    # 3. Split Data (Train 90% / Test 10%) 
    # small test set because we have small data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    print(f"🧠 Data Shape: {X.shape}")
    print(f"🧠 Labels Shape: {y.shape}")
    
    # 4. Build & Train
    model = build_model(X.shape[1:], len(classes))
    
    # TensorBoard for visualization
    tb_callback = TensorBoard(log_dir=LOG_DIR)
    
    model.fit(X_train, y_train, epochs=EPOCHS, callbacks=[tb_callback], validation_data=(X_test, y_test))
    
    # 5. Save Model
    print("💾 Saving Model...")
    model.save(os.path.join(MODELS_PATH, 'tacit_brain.h5'))
    model.save(os.path.join(MODELS_PATH, 'tacit_brain.keras'))
    
    # Save the class labels for inference later
    import pickle
    with open(os.path.join(MODELS_PATH, 'classes.pkl'), 'wb') as f:
        pickle.dump(classes, f)
        
    print("✅ Training Complete. Model saved to training/models/")

if __name__ == "__main__":
    main()