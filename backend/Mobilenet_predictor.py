import os
import numpy as np
from collections import defaultdict
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
)  # ✅ IMPORTANT for MobileNetV2

# ----------- Configuration -------------
MODEL_PATH = r"S:\Personal Projects\Marrow Insight\backend\ml_models\final_mobilenet_model.keras"  # ✅ Adjust path as needed
IMAGE_SIZE = (224, 224)  # ✅ MobileNetV2 uses 224x224 input size

# Class labels used during training (same order as in training script)
class_labels = ["ART", "BLA", "EBO", "LYT", "NGS", "PMO"]

# ----------- Load Model ----------------
print("📦 Loading MobileNetV2 model...")
model = load_model(MODEL_PATH)


# ----------- Preprocessing Function ----
def preprocess_image(img_path, target_size=IMAGE_SIZE):
    """
    Preprocesses image for MobileNetV2 model with proper normalization.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # ✅ USE MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ----------- Single Image Prediction Function ----
def mobilenet_predict_single_image(image_path):
    """
    Predicts the class of a single image using MobileNetV2 model.

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
    Predicts classes for all images in a directory using MobileNetV2 model.

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

    print("🔍 Predicting images with MobileNetV2...")
    for filename in os.listdir(image_directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(image_directory, filename)
            result = mobilenet_predict_single_image(file_path)

            if "error" not in result:
                results.append(result)
                class_frequencies[result["predicted_class"]] += 1
                print(
                    f"{filename} → {result['predicted_class']} ({result['confidence'] * 100:.2f}%)"
                )
            else:
                print(f"Error with {filename}: {result['error']}")

    return results, class_frequencies


# ----------- Model Comparison Function ----
def compare_predictions(image_path, models_dict):
    """
    Compare predictions from multiple models on the same image.

    Args:
        image_path (str): Path to the image file
        models_dict (dict): Dictionary with model names as keys and prediction functions as values
                           e.g., {"MobileNet": mobilenet_predict_func, "ResNet": resnet_predict_func}

    Returns:
        dict: Comparison results from all models
    """
    comparison_results = {}

    for model_name, predict_func in models_dict.items():
        result = predict_func(image_path)
        comparison_results[model_name] = result

    return comparison_results


# ----------- Example Usage -------------
if __name__ == "__main__":
    # Example 1: Predict single image
    single_image_path = (
        "D:/test_images/sample_image.jpg"  # Replace with your image path
    )
    result = mobilenet_predict_single_image(single_image_path)

    if "error" not in result:
        print(f"Predicted Class: {result['cell_type']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")

        for class_name, prob in result["all_probabilities"].items():
            print(f"  {class_name}: {prob * 100:.2f}%")
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
        # df.to_csv("mobilenet_batch_predictions.csv", index=False)
        # print(f"\n✅ Predictions saved to 'mobilenet_batch_predictions.csv'")

        # Display summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"Total images processed: {len(results)}")
        print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.4f}")
        print(f"Highest confidence: {np.max([r['confidence'] for r in results]):.4f}")
        print(f"Lowest confidence: {np.min([r['confidence'] for r in results]):.4f}")
    else:
        print("No valid images found or processed.")

    # Example 3: Advanced - Model comparison (if you have multiple models)
    """
    # Uncomment this section if you want to compare with other models
    from resnet_predictor import predict_single_image as resnet_predict
    from inception_predictor import predict_single_image as inception_predict
    
    models_comparison = {
        "MobileNetV2": predict_single_image,
        "ResNet": resnet_predict,
        "Inception": inception_predict
    }
    
    comparison = compare_predictions(single_image_path, models_comparison)
    
    print(f"\n🔬 Model Comparison for {os.path.basename(single_image_path)}:")
    for model_name, result in comparison.items():
        if "error" not in result:
            print(f"{model_name}: {result['predicted_class']} ({result['confidence']*100:.2f}%)")
        else:
            print(f"{model_name}: {result['error']}")
    """
