import sys
import os
import json
import pickle
from unittest.mock import MagicMock

# --- THE FIX: TRICK WINDOWS ---
sys.modules["tensorflow_decision_forests"] = MagicMock()
sys.modules["tensorflow_decision_forests.keras"] = MagicMock()

import tensorflow as tf
import tensorflowjs as tfjs

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "training", "models", "tacit_transformer.h5")
CLASSES_PKL = os.path.join(BASE_DIR, "training", "models", "classes.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "client", "public", "model")
CLASSES_JSON = os.path.join(OUTPUT_PATH, "classes.json")

def main():
    print(f"📂 Loading Keras model from: {MODEL_PATH}")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except OSError:
        print("❌ Error: Could not find the model file. Did you run train_transformer.py?")
        return

    print(f"🚀 Converting to TensorFlow.js format...")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Convert Model
    tfjs.converters.save_keras_model(model, OUTPUT_PATH)
    print(f"✅ SUCCESS! Model saved to: {OUTPUT_PATH}")

    # Convert Classes to JSON
    if os.path.exists(CLASSES_PKL):
        with open(CLASSES_PKL, 'rb') as f:
            classes = pickle.load(f)
        with open(CLASSES_JSON, 'w') as f:
            json.dump(classes, f)
        print(f"✅ SUCCESS! Classes exported to: {CLASSES_JSON}")
    else:
        print("⚠️ Warning: classes.pkl not found. Skipping JSON export.")

if __name__ == "__main__":
    main()