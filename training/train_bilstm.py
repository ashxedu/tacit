import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import pickle

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODELS_PATH = os.path.join(BASE_DIR, "training", "models")
LOG_DIR = os.path.join(BASE_DIR, "training", "logs")

EPOCHS = 150
BATCH_SIZE = 8 # Lower batch size for high-quality, small dataset

def load_data():
    sequences, labels = [], []
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print("❌ Error: Processed data folder missing.")
        return [], [], []

    file_list = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
    
    if len(file_list) == 0:
         print("❌ Error: No .npy files found. Did you run process_manual.py?")
         return [], [], []

    # Extract labels from filenames (e.g., "welcome_55449.npy" -> "welcome")
    all_labels = sorted(list(set([f.split('_')[0] for f in file_list])))
    label_map = {label: num for num, label in enumerate(all_labels)}
    
    print(f"✅ Found {len(all_labels)} classes: {all_labels}")
    
    for filename in file_list:
        word = filename.split('_')[0]
        res = np.load(os.path.join(DATA_PATH, filename))
        
        # Standardize to 30 frames
        # If video is short, pad with zeros. If long, cut it.
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
    
    # --- BiLSTM LAYER 1 ---
    # Bidirectional allows the model to see the "future" context of the gesture
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # --- BiLSTM LAYER 2 ---
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))

    # --- DECISION ---
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    if not os.path.exists(MODELS_PATH): os.makedirs(MODELS_PATH)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    
    X, y, classes = load_data()
    
    if len(classes) == 0: return

    y = to_categorical(y).astype(int)
    
    # Split data (10% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    print(f"🧠 Training on {X.shape[0]} samples with BiLSTM architecture...")

    model = build_bilstm_model(X.shape[1:], len(classes))
    
    # Early Stopping to prevent overfitting (Stops if accuracy doesn't improve for 30 epochs)
    early = EarlyStopping(monitor='val_categorical_accuracy', patience=30, restore_best_weights=True)
    tb = TensorBoard(log_dir=LOG_DIR)
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              callbacks=[tb, early], validation_data=(X_test, y_test))
    
    # Save Model
    model.save(os.path.join(MODELS_PATH, 'tacit_brain.h5'))
    
    # Save Class Names
    with open(os.path.join(MODELS_PATH, 'classes.pkl'), 'wb') as f:
        pickle.dump(classes, f)
        
    print(f"✅ Training Complete. Class order: {classes}")

if __name__ == "__main__":
    main()