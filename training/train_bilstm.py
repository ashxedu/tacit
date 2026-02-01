import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODELS_PATH = os.path.join(BASE_DIR, "training", "models")
LOG_DIR = os.path.join(BASE_DIR, "training", "logs")

EPOCHS = 200        # High epochs (EarlyStopping will save us)
BATCH_SIZE = 8      # Low batch size = More updates = Better learning on small data

def load_data():
    sequences, labels = [], []
    
    if not os.path.exists(DATA_PATH): return [], [], []
    file_list = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
    if len(file_list) == 0: return [], [], []

    # Extract labels and sort alphabetically
    all_labels = sorted(list(set([f.split('_')[0] for f in file_list])))
    label_map = {label: num for num, label in enumerate(all_labels)}
    
    print(f"✅ Found {len(all_labels)} classes: {all_labels}")
    
    for filename in file_list:
        word = filename.split('_')[0]
        res = np.load(os.path.join(DATA_PATH, filename))
        
        # Standardize length to 30 frames
        target_len = 30
        if len(res) > target_len: 
            res = res[:target_len]
        elif len(res) < target_len: 
            res = np.concatenate((res, np.zeros((target_len - len(res), 258))))
            
        sequences.append(res)
        labels.append(label_map[word])
        
    return np.array(sequences), np.array(labels), all_labels

def build_bilstm_model(input_shape, num_classes):
    model = Sequential()
    
    # --- BiLSTM BEAST MODE ---
    # Layer 1: Reads forward and backward
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3)) # Slightly higher dropout for better generalization
    
    # Layer 2: Condenses information
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))

    # Layer 3: Decision
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    if not os.path.exists(MODELS_PATH): os.makedirs(MODELS_PATH)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    
    X, y, classes = load_data()
    if len(classes) == 0: 
        print("❌ No data found. Run process_manual.py first.")
        return

    y = to_categorical(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # 10% for validation
    
    print(f"🧠 Training on {X.shape[0]} samples...")

    model = build_bilstm_model(X.shape[1:], len(classes))
    
    # --- CALLBACKS ---
    # 1. TensorBoard: Visualization
    tb = TensorBoard(log_dir=LOG_DIR)
    
    # 2. EarlyStopping: Stop if not improving for 35 epochs
    early = EarlyStopping(monitor='val_categorical_accuracy', patience=35, restore_best_weights=True)
    
    # 3. ReduceLROnPlateau: If stuck, slow down learning rate to find the minimum
    lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              callbacks=[tb, early, lr_decay], 
              validation_data=(X_test, y_test))
    
    # Save
    model.save(os.path.join(MODELS_PATH, 'tacit_brain.h5'))
    with open(os.path.join(MODELS_PATH, 'classes.pkl'), 'wb') as f:
        pickle.dump(classes, f)
        
    print(f"✅ Training Complete. Class order: {classes}")

if __name__ == "__main__":
    main()