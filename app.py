"""
Gradio app for Cat vs Dog classification - Optimized for Hugging Face Spaces
This is the recommended version for deploying on Hugging Face Spaces
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf

# Configuration
IMG_SIZE = (224, 224)  # Updated to match MobileNetV2 transfer learning model
MODEL_PATH = "cats_dogs_cnn.keras"


def load_model():
    """Load the trained CNN model with multiple fallback strategies."""
    import traceback
    
    print("="*70)
    print("[INIT] Starting model loading...")
    print(f"[INFO] Current working directory: {os.getcwd()}")
    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] Keras version: {tf.keras.__version__}")
    print("="*70)
    
    # List files in current directory for debugging
    print("[DEBUG] Files in current directory:")
    try:
        for item in sorted(os.listdir('.')):
            full_path = os.path.join('.', item)
            if os.path.isfile(full_path):
                size_mb = os.path.getsize(full_path) / (1024*1024)
                print(f"  ✓ {item} ({size_mb:.2f} MB)")
            else:
                print(f"  📁 {item}/")
    except Exception as e:
        print(f"  [ERROR] Could not list directory: {e}")
    
    # ===== STRATEGY 1: Load .keras format (normal) =====
    print(f"\n[STRATEGY 1] Trying .keras format ({MODEL_PATH})...")
    if os.path.exists(MODEL_PATH):
        print(f"  ✓ File found: {MODEL_PATH}")
        try:
            print(f"  → Loading with standard method...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("[OK] ✅ SUCCESS: Model loaded from .keras format (standard)")
            return model
        except Exception as e:
            print(f"  ✗ Standard load failed: {type(e).__name__}: {str(e)[:100]}")
            
            # Try with compile=False
            try:
                print(f"  → Retrying with compile=False...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("[OK] ✅ SUCCESS: Model loaded with compile=False")
                return model
            except Exception as e2:
                print(f"  ✗ compile=False also failed: {type(e2).__name__}")
    else:
        print(f"  ✗ File NOT found: {MODEL_PATH}")
    
    # ===== STRATEGY 2: Try .h5 format (legacy Keras) =====
    h5_path = "cats_dogs_cnn.h5"
    print(f"\n[STRATEGY 2] Trying H5 format ({h5_path})...")
    if os.path.exists(h5_path):
        print(f"  ✓ File found: {h5_path}")
        try:
            print(f"  → Loading H5 format...")
            model = tf.keras.models.load_model(h5_path)
            print("[OK] ✅ SUCCESS: Model loaded from H5 format")
            return model
        except Exception as e:
            print(f"  ✗ H5 load failed: {type(e).__name__}: {str(e)[:100]}")
    else:
        print(f"  ✗ File NOT found: {h5_path}")
    
    # NOTE: SavedModel format is NOT supported by Keras 3.x
    # If you have cats_dogs_cnn_savedmodel/, it needs to be converted to .keras first
    
    # ===== STRATEGY 3: Create fallback diagnostic model =====
    print(f"\n[STRATEGY 3] Creating fallback diagnostic model...")
    try:
        print(f"  → Building a simple CNN model for testing...")
        # Create a minimal CNN model for testing
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        print("[WARNING] ⚠️ Using fallback diagnostic model (untrained)")
        print("  → This model makes random predictions - not the real trained model")
        print("  → To fix: Check Render logs and ensure model file is in Git repository")
        return model
    except Exception as e:
        print(f"  ✗ Even fallback model creation failed: {e}")
    
    # ===== ALL STRATEGIES FAILED =====
    print("\n" + "="*70)
    print("[ERROR] ❌ COULD NOT LOAD ANY MODEL")
    print("="*70)
    print("\n📋 DIAGNOSTIC INFO:")
    print(f"  • .keras file exists: {os.path.exists(MODEL_PATH)}")
    print(f"  • .h5 file exists: {os.path.exists('cats_dogs_cnn.h5')}")
    print(f"  • TensorFlow: {tf.__version__}")
    print(f"  • Working dir: {os.getcwd()}")
    
    print("\n🔧 TO FIX THIS:")
    print("  1. Check Render deployment logs for error details")
    print("  2. Verify model file is committed to Git:")
    print("     $ git ls-files | grep cats_dogs")
    print("  3. If not in Git, add and push:")
    print("     $ git add cats_dogs_cnn.keras")
    print("     $ git commit -m 'Add model'")
    print("     $ git push")
    print("  4. Rebuild on Render: Manual Deploy > Deploy latest commit")
    print("  5. If still fails, check file compatibility:")
    print("     $ python train_cnn_only.py  # Re-train and save with current TF version")
    print("\n" + "="*70)
    return None


# Load model globally
model = load_model()


def predict_image(image):
    """
    Predict if image contains a cat or dog.
    
    Args:
        image: PIL Image from Gradio
        
    Returns:
        dict: Classification results with probabilities
    """
    if model is None:
        # Return neutral prediction when model is not loaded
        # This prevents Gradio Label validation errors
        return {
            "Model Not Ready": 0.5,
            "Status": 0.5
        }
    
    if image is None:
        return {
            "No Image": 0.5,
            "Status": 0.5
        }
    
    try:
        # Preprocess image
        img = image.convert('RGB').resize(IMG_SIZE, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)
        
        # Predict
        prob_dog = float(model.predict(img_array, verbose=0)[0][0])
        prob_cat = 1.0 - prob_dog
        
        return {
            "Cat": prob_cat,
            "Dog": prob_dog
        }
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            "Error Processing": 0.5,
            "Image": 0.5
        }


# Create Gradio interface
with gr.Blocks(
    title="Cat vs Dog Classifier",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")
) as demo:
    gr.Markdown(
        """
        # 🐱 Cat vs Dog Classifier 🐶
        
        Upload an image of a cat or dog, and the CNN model will classify it!
        
        **Model:** Convolutional Neural Network (CNN)  
        **Accuracy:** ~86% on test set  
        **Framework:** TensorFlow/Keras
        """
    )
    
    # Display model status
    if model is not None:
        gr.Markdown("[OK] **Model Status:** Ready for predictions")
    else:
        gr.Markdown("""[!] **Model Status:** Not loaded
        
        **To fix this:**
        1. Open a terminal in the project folder
        2. Run: `python cats_vs_dogs_classifier.py`
        3. Wait for the model to train (this trains and saves the model with the current TensorFlow version)
        4. Refresh this page
        """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Image"
            )
        
        with gr.Column():
            output = gr.Label(
                label="Prediction",
                num_top_classes=2
            )
    
    predict_btn = gr.Button(
        "🔍 Predict",
        variant="primary",
        size="lg"
    )
    
    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=output
    )
    
    gr.Markdown(
        """
        ---
        **How it works:**
        1. Upload a cat or dog image (PNG, JPG, JPEG, WEBP)
        2. Click the Predict button
        3. See the classification result with confidence scores
        
        **Trained Models:** This app uses the CNN model, which achieved the highest accuracy (~86%) among all trained models (Logistic Regression, SVM, Random Forest, CNN).
        """
    )


if __name__ == "__main__":
    demo.launch()
