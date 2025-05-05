import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import logging
import warnings

# Suppress TensorFlow and CUDA logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # Suppress TensorFlow logs (0 = all logs, 1 = warnings, 2 = errors, 3 = fatal)
)
logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Suppress TensorFlow logger
warnings.filterwarnings("ignore")  # Suppress Python warnings


class PlantDiseasePredictionService:
    def __init__(self, model_path, class_names, image_size=(256, 256)):
        """
        Initializes the PlantDiseasePredictionService by loading the model and setting up configurations.

        Args:
            model_path (str): Path to the trained model file (.h5).
            class_names (tuple): Tuple of class names in the same order as the model's output.
            image_size (tuple): The input size expected by the model.
        """
        self.model_path = model_path
        self.class_names = class_names
        self.image_size = image_size

        # Force TensorFlow to use CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Load the model
        self._load_model()

    def _load_model(self):
        """Loads the TensorFlow model from the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        try:
            self.model = load_model(self.model_path)
            # self.model.summary()  # Optional: Print model summary
            # print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Error loading model: {e}. Ensure the file is not corrupted and dependencies are installed."
            )

    def predict(self, image_path):
        """
        Predicts the plant disease for a given image.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            dict: A dictionary containing prediction results:
                  'plant_name': The name of the plant.
                  'disease_name': The name of the disease.
                  'confidence': The confidence score (probability) for the prediction.
                  'inference_time_ms': Time taken for inference in milliseconds.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")

        try:
            # Load and preprocess the image
            opencv_image = cv2.imread(image_path)
            if opencv_image is None:
                raise ValueError(f"Could not read image file: {image_path}")

            resized_image = cv2.resize(opencv_image, self.image_size)
            normalized_image = resized_image / 255.0  # Normalize to [0, 1]
            input_image = np.expand_dims(normalized_image, axis=0).astype(np.float32)

            # Perform inference
            start_time = time.time()
            predictions = self.model.predict(input_image)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Get the predicted class and confidence
            predicted_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))

            if predicted_index < 0 or predicted_index >= len(self.class_names):
                raise ValueError(
                    f"Predicted index ({predicted_index}) out of bounds for class names."
                )

            predicted_class_name = self.class_names[predicted_index]
            parts = predicted_class_name.split("-")
            plant_name = parts[0] if len(parts) > 0 else "Unknown"
            disease_name = parts[1] if len(parts) > 1 else "Unknown"

            return {
                "plant_name": plant_name,
                "disease_name": disease_name,
                "confidence": confidence,
                "inference_time_ms": inference_time,
            }

        except Exception as e:
            raise RuntimeError(f"An error occurred during prediction: {str(e)}")


# ---------------------- Singleton Instance ----------------------
# Create a singleton instance of the PlantDiseasePredictionService
plant_disease_prediction_service = PlantDiseasePredictionService(
    model_path="models/weights/plant_disease_model.h5",
    class_names=("Tomato-Bacterial_spot", "Potato-Early_blight", "Corn-Common_rust"),
)
