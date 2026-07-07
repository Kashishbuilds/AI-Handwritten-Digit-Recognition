"""
predict.py
----------
Handles loading the trained model (model.h5) and running predictions.

This module is used by app.py so that the Streamlit app never has to
retrain the model - it just loads the saved weights and predicts.

You can also run this file directly from the command line to test a
prediction on a single image file:

    python predict.py path/to/digit.png
"""

import sys
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from utils import preprocess_image, prepare_for_model

MODEL_PATH = "model.h51"


def load_trained_model(model_path=MODEL_PATH):
    """
    Load the trained CNN model from disk.

    Raises
    ------
    FileNotFoundError
        If model.h5 does not exist (user needs to run train_model.py first).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find '{model_path}'. "
            f"Please run 'python train_model.py' first to train and save the model."
        )
    try:
        model = load_model(model_path)
        return model
    except Exception as error:
        raise RuntimeError(f"Failed to load model from {model_path}: {error}")


def predict_digit(model, image):
    """
    Predict the digit shown in `image` using the given trained model.

    Parameters
    ----------
    model : keras.Model
        The loaded CNN model.
    image : PIL.Image.Image or np.ndarray
        Raw input image (from upload or drawing canvas).

    Returns
    -------
    dict with keys:
        "digit"        -> predicted digit (int, 0-9)
        "confidence"   -> confidence of the top prediction (float, 0-100)
        "probabilities"-> list of 10 floats, probability for each digit
        "processed_image" -> the 28x28 preprocessed image (for display/debug)
    """
    try:
        # Step 1: Preprocess the raw image into a clean 28x28 normalized array
        processed = preprocess_image(image)

        # Step 2: Reshape for the model (1, 28, 28, 1)
        model_input = prepare_for_model(processed)

        # Step 3: Run prediction. Output is a softmax probability array of shape (1, 10)
        probabilities = model.predict(model_input, verbose=0)[0]

        # Step 4: Extract the predicted digit and its confidence
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100

        return {
            "digit": predicted_digit,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "processed_image": processed,
        }

    except Exception as error:
        raise RuntimeError(f"Error during prediction: {error}")


def main():
    """Command line entry point for quick testing."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        sys.exit(1)

    try:
        model = load_trained_model()
        image = Image.open(image_path)
        result = predict_digit(model, image)

        print("\n--- Prediction Result ---")
        print(f"Predicted Digit : {result['digit']}")
        print(f"Confidence      : {result['confidence']:.2f}%")
        print("Class Probabilities:")
        for digit, prob in enumerate(result["probabilities"]):
            print(f"  {digit}: {prob * 100:.2f}%")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
