import sys
import os
from unittest.mock import MagicMock

# --- THE FIX: TRICK WINDOWS ---
# We inject a fake 'tensorflow_decision_forests' module into Python's memory.
# This stops tensorflowjs from crashing when it tries to import it.
sys.modules["tensorflow_decision_forests"] = MagicMock()
sys.modules["tensorflow_decision_forests.keras"] = MagicMock()

# --- NOW WE CAN IMPORT SAFELY ---
import tensorflow as tf
import tensorflowjs as tfjs

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "training", "models", "tacit_brain.h5")
OUTPUT_PATH = os.path.join(BASE_DIR, "client", "public", "model")

def main():
    print(f"📂 Loading Keras model from: {MODEL_PATH}")
    
    # Load the trained model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except OSError:
        print("❌ Error: Could not find the model file. Did you run train.py?")
        return

    print(f"🚀 Converting to TensorFlow.js format...")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Convert and Save
    tfjs.converters.save_keras_model(model, OUTPUT_PATH)
    
    print(f"✅ SUCCESS! Model saved to: {OUTPUT_PATH}")
    print("   You should see 'model.json' and binary shard files there.")

if __name__ == "__main__":
    main()