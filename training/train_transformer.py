import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    GlobalAveragePooling1D, Add, Conv1D, Dot, Activation
)
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODELS_PATH = os.path.join(BASE_DIR, "training", "models")
LOG_DIR = os.path.join(BASE_DIR, "training", "logs_transformer")

EPOCHS = 200
BATCH_SIZE = 8
MAX_FRAMES = 30
BASE_FEATURES = 226 # Pruned lower-body features
TOTAL_FEATURES = BASE_FEATURES * 2 # 452 (Positions + Velocities)

def normalize_and_prune(frame):
    # 1. Isolate components (Keep only upper body)
    pose_pruned = frame[0:100].copy()
    lh = frame[132:195].copy()
    rh = frame[195:258].copy()

    # 2. Chest Anchor
    ls = np.array([frame[44], frame[45], frame[46]])
    rs = np.array([frame[48], frame[49], frame[50]])
    anchor = (ls + rs) / 2.0
    
    # 3. Shoulder Ruler
    shoulder_dist = np.linalg.norm(ls - rs)
    if shoulder_dist < 1e-5:
        shoulder_dist = 1.0 
        
    pruned_frame = np.concatenate([pose_pruned, lh, rh])

    # 4. Normalize EVERYTHING to the Chest
    for i in range(0, 226, 4 if i < 100 else 3):
        # Skip visibility flag for pose (every 4th item)
        if i < 100 and (i + 3) % 4 == 3: continue 
        
        if pruned_frame[i] == 0 and pruned_frame[i+1] == 0: continue
        pruned_frame[i]   = (pruned_frame[i]   - anchor[0]) / shoulder_dist
        pruned_frame[i+1] = (pruned_frame[i+1] - anchor[1]) / shoulder_dist
        pruned_frame[i+2] = (pruned_frame[i+2] - anchor[2]) / shoulder_dist

    return pruned_frame

def load_data():
    sequences, labels = [], []
    if not os.path.exists(DATA_PATH): 
        return [], [], []
    
    file_list = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
    if not file_list: return [], [], []
    
    all_labels = sorted(list(set([f.split('_')[0] for f in file_list])))
    label_map = {label: num for num, label in enumerate(all_labels)}
    print(f"✅ Found {len(all_labels)} classes: {all_labels}")
    
    for filename in file_list:
        word = filename.split('_')[0]
        res = np.load(os.path.join(DATA_PATH, filename))
        
        if len(res) > MAX_FRAMES: res = res[:MAX_FRAMES]
        elif len(res) < MAX_FRAMES: res = np.concatenate((res, np.zeros((MAX_FRAMES - len(res), 258))))
            
        # Apply the Dual-Anchor Geometry Filter to every frame
        normalized_res = np.array([normalize_and_prune(frame) for frame in res])
        
        # Calculate Velocities on the normalized data
        velocities = np.diff(normalized_res, axis=0)
        velocities = np.vstack([np.zeros((1, BASE_FEATURES)), velocities])
        res_combined = np.concatenate((normalized_res, velocities), axis=1)
            
        sequences.append(res_combined)
        labels.append(label_map[word])
        
    return np.array(sequences), np.array(labels), all_labels

def build_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    
    query = Dense(64)(x)
    key = Dense(64)(x)
    value = Dense(64)(x)
    
    attention_scores = Dot(axes=[2, 2])([query, key])
    attention_weights = Activation('softmax')(attention_scores)
    attention_output = Dot(axes=[2, 1])([attention_weights, value])
    
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    ffn_output = Dense(128, activation="relu")(x)
    ffn_output = Dense(64)(ffn_output)
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    if not os.path.exists(MODELS_PATH): os.makedirs(MODELS_PATH)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    X, y, classes = load_data()
    if len(classes) == 0: return

    y = tf.keras.utils.to_categorical(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    print(f"🧠 Training Tacit on {X.shape[0]} normalized samples (Dual-Anchor 452 features)...")

    model = build_transformer_model((MAX_FRAMES, TOTAL_FEATURES), len(classes))
    tb = TensorBoard(log_dir=LOG_DIR)
    early = EarlyStopping(monitor='val_categorical_accuracy', patience=40, restore_best_weights=True)
    lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tb, early, lr_decay], validation_data=(X_test, y_test))
    model.save(os.path.join(MODELS_PATH, 'tacit_transformer.h5'))
    with open(os.path.join(MODELS_PATH, 'classes.pkl'), 'wb') as f: pickle.dump(classes, f)
    print(f"✅ Training Complete. Class order: {classes}")

if __name__ == "__main__":
    main()