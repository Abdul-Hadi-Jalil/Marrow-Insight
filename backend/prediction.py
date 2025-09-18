import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ----------- Configuration -------------
MODEL_PATH = (
    "S:/Personal Projects/Marrow Insight/backend/ml_models/final_model_inception.keras"
)
# MODEL_PATH = "D:/Semesters Material/FYP/marrowInsight/Code/final_model_resnet_finetuned.keras"
IMAGE_SIZE = (150, 150)

# The class order used during training
class_labels = ["ART", "BLA", "EBO", "LYT", "NGS", "PMO"]

# ----------- Load Model ----------------
model = load_model(MODEL_PATH)


# ----------- Preprocessing Function ----
def preprocess_image(img_path, target_size=IMAGE_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ----------- Single Image Prediction Function ----
def predict_single_image(image_path):
    """
    Predicts the class of a single image.

    Args:
        image_path (str): Path to the image file

    Returns:
        dict: Dictionary containing filename, predicted_class, confidence, and all_probabilities
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}

        # Check if file has valid image extension
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            return {
                "error": f"Invalid image format. Supported formats: .jpg, .jpeg, .png"
            }

        # Preprocess the image
        img = preprocess_image(image_path)

        # Make prediction
        pred = model.predict(img, verbose=0)
        pred_class = class_labels[np.argmax(pred)]
        confidence = np.max(pred)

        return {
            "cell_type": pred_class,
            "confidence": round(float(confidence), 4),
        }

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}


# ----------- Example Usage -------------
if __name__ == "__main__":
    # Example 1: Predict single image
    single_image_path = r"S:\Final year Project dataset\train/NGS\NGS_07001.jpg"  # Replace with your image path
    result = predict_single_image(single_image_path)

    if "error" not in result:
        print("\n🔍 Single Image Prediction:")

        print(f"Predicted Class: {result['cell_type']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
    else:
        print(f"{result['error']}")
