import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODELS_PATH = os.path.join(BASE_DIR, "training", "models")
LOG_DIR = os.path.join(BASE_DIR, "training", "logs")

MIN_SAMPLES_REQUIRED = 30 
EPOCHS = 150   
BATCH_SIZE = 16 

def load_data():
    sequences, labels = [], []
    file_list = os.listdir(DATA_PATH)
    
    class_counts = {}
    for filename in file_list:
        if not filename.endswith(".npy"): continue
        word = filename.split('_')[0]
        class_counts[word] = class_counts.get(word, 0) + 1
        
    valid_classes = sorted([w for w, c in class_counts.items() if c >= MIN_SAMPLES_REQUIRED])
    label_map = {label: num for num, label in enumerate(valid_classes)}
    
    print(f"✅ Training on {len(valid_classes)} classes: {valid_classes}")
    
    for filename in file_list:
        word = filename.split('_')[0]
        if word not in valid_classes: continue
        
        path = os.path.join(DATA_PATH, filename)
        res = np.load(path)
        
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
    model = Sequential()
    
    # 1. Feature Extraction (CNN)
    # We keep this to detect "Motion" vs "Static"
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization()) 

    # 2. Sequence Learning (LSTM)
    # Simpler: One strong LSTM layer instead of two weak ones
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2)) # Reduced from 0.4 to 0.2 (Let it learn!)

    # 3. Decision Making
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2)) # Safety rail
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    if not os.path.exists(MODELS_PATH): os.makedirs(MODELS_PATH)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    X, y, classes = load_data()
    
    if len(classes) == 0:
        print("❌ Error: No valid classes found.")
        return

    y = to_categorical(y).astype(int)
    # Reset test size to standard 0.1 to give training more data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) 
    
    print(f"🧠 Data Shape: {X.shape}")
    
    model = build_model(X.shape[1:], len(classes))
    
    # Patience increased slightly to allow for "Learning Plateaus"
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=20, restore_best_weights=True)
    tb_callback = TensorBoard(log_dir=LOG_DIR)
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              callbacks=[tb_callback, early_stopping], 
              validation_data=(X_test, y_test))
    
    print("💾 Saving Model...")
    model.save(os.path.join(MODELS_PATH, 'tacit_brain.h5'))
    
    import pickle
    with open(os.path.join(MODELS_PATH, 'classes.pkl'), 'wb') as f:
        pickle.dump(classes, f)
        
    print("✅ Training Complete.")
    print(f"📋 FINAL CLASS ORDER: {classes}")

if __name__ == "__main__":
    main()