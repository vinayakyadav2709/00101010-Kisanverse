import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import time


# ---------------------- Model Definition ----------------------
class SoilClassifier(nn.Module):
    def __init__(self, model_name="resnet50", num_classes=11, pretrained=False):
        super(SoilClassifier, self).__init__()

        if model_name == "resnet50":
            base_model = models.resnet50(weights=None)
            num_features = base_model.fc.in_features
            base_model.fc = nn.Identity()  # Remove the original classifier head
            self.features = base_model

            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------- Soil Prediction Class ----------------------
class SoilPredictionService:
    def __init__(self, model_path, device=None):
        """
        Initializes the SoilPredictionService by loading the model and setting up transformations.

        Args:
            model_path (str): Path to the trained model checkpoint (.pth file).
            device (torch.device, optional): Device to run inference on (CPU or CUDA). Defaults to auto-detect.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path

        # Load the model and checkpoint
        self._load_model()

        # Define image transformations (must match those used during training)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self):
        """Loads the model and its weights from the checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        num_classes = len(self.class_names)

        self.model = SoilClassifier(model_name="resnet50", num_classes=num_classes).to(
            self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, image_path):
        """
        Predicts the soil type for a given image.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            dict: A dictionary containing prediction results:
                  'predicted_class': The name of the predicted class.
                  'confidence': The confidence score (probability) for the predicted class.
                  'all_probabilities': A list of tuples (class_name, probability) sorted descending.
                  'inference_time_ms': Time taken for inference in milliseconds.
                  'error': An error message if prediction failed, None otherwise.
        """
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Perform inference
            start_time = time.time()
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]

            inference_time = time.time() - start_time

            # Get the predicted class index and confidence
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class_name = self.class_names[predicted_idx.item()]

            # Get all class probabilities and sort them
            probs_dict = {
                self.class_names[i]: prob.item() for i, prob in enumerate(probabilities)
            }
            sorted_probs = sorted(
                probs_dict.items(), key=lambda item: item[1], reverse=True
            )

            return {
                "predicted_class": predicted_class_name,
                "confidence": confidence.item(),
                "all_probabilities": sorted_probs,
                "inference_time_ms": inference_time * 1000,
                "error": None,
            }

        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at {image_path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during prediction: {str(e)}") from e


# ---------------------- Singleton Instance ----------------------
# Create a singleton instance of the SoilPredictionService
soil_prediction_service = SoilPredictionService(
    model_path="models/weights/best_resnet50_soil_model.pth"  # Replace with the actual path
)
