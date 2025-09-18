import os
import numpy as np
from collections import defaultdict
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import (
    preprocess_input,
)  # ✅ IMPORTANT for ResNet

# ----------- Configuration -------------
# MODEL_PATH = "D:/Semesters Material/FYP/marrowInsight/Code/final_model_resnet_finetuned.keras"  # ✅ Correct model path
IMAGE_SIZE = (150, 150)
MODEL_PATH = (
    r"S:\Personal Projects\Marrow Insight\backend\ml_models\best_model_resnet.keras"
)
# Class labels used during training
class_labels = ["ART", "BLA", "EBO", "LYT", "NGS", "PMO"]

# ----------- Load Model ----------------
print("📦 Loading ResNet model...")
model = load_model(MODEL_PATH)


# ----------- Preprocessing Function ----
def preprocess_image(img_path, target_size=IMAGE_SIZE):
    """
    Preprocesses image for ResNet model with proper normalization.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # ✅ USE THE SAME as in training
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ----------- Single Image Prediction Function ----
def resnet_predict_single_image(image_path):
    """
    Predicts the class of a single image using ResNet model.

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

        # Get filename from path
        filename = os.path.basename(image_path)

        # Create probability dictionary for all classes
        all_probabilities = {
            class_labels[i]: round(float(pred[0][i]), 4)
            for i in range(len(class_labels))
        }

        return {
            "cell_type": pred_class,
            "confidence": round(float(confidence), 4),
        }

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}


# ----------- Batch Prediction Function (Optional) ----
def predict_batch_images(image_directory):
    """
    Predicts classes for all images in a directory using ResNet model.

    Args:
        image_directory (str): Path to directory containing images

    Returns:
        tuple: (results_list, class_frequencies_dict)
    """
    results = []
    class_frequencies = defaultdict(int)

    if not os.path.exists(image_directory):
        print(f"Directory not found: {image_directory}")
        return results, class_frequencies

    print("🔍 Predicting images...")
    for filename in os.listdir(image_directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(image_directory, filename)
            result = resnet_predict_single_image(file_path)

            if "error" not in result:
                results.append(result)
                class_frequencies[result["predicted_class"]] += 1
                print(
                    f"{filename} → {result['predicted_class']} ({result['confidence'] * 100:.2f}%)"
                )
            else:
                print(f"Error with {filename}: {result['error']}")

    return results, class_frequencies


# ----------- Example Usage -------------
if __name__ == "__main__":
    # Example 1: Predict single image
    single_image_path = (
        "D:/test_images/sample_image.jpg"  # Replace with your image path
    )
    result = resnet_predict_single_image(single_image_path)

    if "error" not in result:
        print(f"Predicted Class: {result['cell_type']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")

    else:
        print(f"❌ {result['error']}")

    # Example 2: Batch prediction (optional)
    IMAGE_DIR = "D:/test_images"  # Replace with your test folder
    print(f"\n📁 Batch Prediction from: {IMAGE_DIR}")
    results, class_frequencies = predict_batch_images(IMAGE_DIR)

    if results:
        print(f"\n📊 Class Frequencies:")
        for cls in class_labels:
            print(f"{cls}: {class_frequencies[cls]}")

        # Save to CSV
        # df = pd.DataFrame(results)
        # df.to_csv("resnet_batch_predictions.csv", index=False)
        # print(f"\n✅ Predictions saved to 'resnet_batch_predictions.csv'")
    else:
        print("No valid images found or processed.")
