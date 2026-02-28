import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D, Add, Conv1D
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
NUM_FEATURES = 258

def load_data():
    sequences, labels = [], []
    if not os.path.exists(DATA_PATH): 
        return [], [], []
    
    file_list = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
    if not file_list:
        return [], [], []
    
    all_labels = sorted(list(set([f.split('_')[0] for f in file_list])))
    label_map = {label: num for num, label in enumerate(all_labels)}
    print(f"✅ Found {len(all_labels)} classes: {all_labels}")
    
    for filename in file_list:
        word = filename.split('_')[0]
        res = np.load(os.path.join(DATA_PATH, filename))
        
        # Standardize length to 30 frames
        if len(res) > MAX_FRAMES: 
            res = res[:MAX_FRAMES]
        elif len(res) < MAX_FRAMES: 
            res = np.concatenate((res, np.zeros((MAX_FRAMES - len(res), NUM_FEATURES))))
            
        sequences.append(res)
        labels.append(label_map[word])
        
    return np.array(sequences), np.array(labels), all_labels

def build_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # 1. Convolutional Positional Encoding Layer
    # Projects the 258 features down to 64 and blends adjacent frames.
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    
    # 2. Transformer Block
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward Network
    ffn_output = Dense(128, activation="relu")(x)
    ffn_output = Dense(64)(ffn_output) # Must match the 64 filters from Conv1D to Add()
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # 3. Classification Head
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
    if len(classes) == 0: 
        print("❌ No processed data found. Did you run process_manual.py?")
        return

    y = tf.keras.utils.to_categorical(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    print(f"🧠 Training Tacit Transformer on {X.shape[0]} samples...")

    model = build_transformer_model((MAX_FRAMES, NUM_FEATURES), len(classes))
    model.summary()
    
    tb = TensorBoard(log_dir=LOG_DIR)
    early = EarlyStopping(monitor='val_categorical_accuracy', patience=40, restore_best_weights=True)
    lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    
    # Train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              callbacks=[tb, early, lr_decay], 
              validation_data=(X_test, y_test))
    
    # model and classes
    model.save(os.path.join(MODELS_PATH, 'tacit_transformer.h5'))
    with open(os.path.join(MODELS_PATH, 'classes.pkl'), 'wb') as f:
        pickle.dump(classes, f)
        
    print(f"✅ Training Complete. Class order: {classes}")

if __name__ == "__main__":
    main()