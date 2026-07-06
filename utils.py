"""
utils.py
--------
Shared helper functions used by both app.py and predict.py.

This module contains:
    1. Image preprocessing functions (grayscale, resize, normalize, invert)
    2. A function to convert a PIL/np image into the shape the CNN expects
    3. A helper to draw a nice probability bar chart with Matplotlib

Keeping these functions in one place means we only have to write and test
the preprocessing logic once, and both the Streamlit app and the command
line predictor stay in sync.
"""

import numpy as np
from PIL import Image
import matplotlib

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. IMAGE PREPROCESSING
# ----------------------------------------------------------------------
def preprocess_image(image):
    """
    Convert any input image (PIL Image, numpy array, or canvas RGBA array)
    into a clean 28x28 grayscale, normalized image ready for the CNN.

    Steps performed:
        1. Convert to a numpy array (if it isn't already one)
        2. Convert to grayscale
        3. Resize to 28x28 (the size MNIST digits use)
        4. Invert colors if the background is white and digit is dark
           (MNIST digits are white strokes on a BLACK background)
        5. Normalize pixel values to the range [0, 1]

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray
        The raw input image (from file upload or drawing canvas).

    Returns
    -------
    np.ndarray
        A (28, 28) float32 array, normalized between 0 and 1.
    """
    try:
        # Step 1: Make sure we are working with a numpy array
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        elif not isinstance(image, np.ndarray):
            raise TypeError(
                "preprocess_image expects a PIL Image or numpy.ndarray"
            )

        # If the canvas gives us an RGBA image, drop the alpha channel
        if image.ndim == 3 and image.shape[2] == 4:
            if cv2 is not None:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                image = np.array(
                    Image.fromarray(image, mode="RGBA").convert("RGB")
                )

        # Step 2: Convert to grayscale (if not already single channel)
        if image.ndim == 3:
            if cv2 is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(
                    "uint8"
                )
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(
                "Unsupported image shape for preprocessing: "
                f"{image.shape}"
            )

        if gray.dtype != np.uint8:
            gray = gray.astype("uint8")

        # Step 3: Resize to 28x28 pixels (MNIST's native size)
        if cv2 is not None:
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            resized = np.array(
                Image.fromarray(gray).resize((28, 28), resample=Image.LANCZOS)
            )

        # Step 4: Decide whether we need to invert colors.
        # MNIST digits are WHITE strokes on a BLACK background.
        # Most uploaded photos / scanned notes are the opposite
        # (dark ink on a light/white background), so we invert them.
        mean_pixel_value = np.mean(resized)
        if mean_pixel_value > 127:
            if cv2 is not None:
                resized = cv2.bitwise_not(resized)
            else:
                resized = 255 - resized

        # Step 5: Normalize pixel values between 0 and 1
        normalized = resized.astype("float32") / 255.0

        return normalized

    except Exception as error:
        raise ValueError(f"Error while preprocessing image: {error}")


def prepare_for_model(processed_image):
    """
    Reshape a (28, 28) normalized image into the (1, 28, 28, 1) shape
    that the CNN model expects as input (batch_size, height, width, channels).

    Parameters
    ----------
    processed_image : np.ndarray
        Output of preprocess_image(), shape (28, 28).

    Returns
    -------
    np.ndarray
        Shape (1, 28, 28, 1), ready to feed into model.predict().
    """
    try:
        if processed_image.ndim == 2:
            return processed_image.reshape(1, 28, 28, 1)
        if processed_image.ndim == 3 and processed_image.shape == (28, 28, 1):
            return processed_image.reshape(1, 28, 28, 1)
        raise ValueError(
            "processed_image must have shape (28, 28) or (28, 28, 1)"
        )
    except Exception as error:
        raise ValueError(f"Error while reshaping image for model: {error}")


# ----------------------------------------------------------------------
# 2. VISUALIZATION HELPERS
# ----------------------------------------------------------------------
def plot_probability_chart(probabilities):
    """
    Create a Matplotlib bar chart showing the predicted probability
    for each digit class (0-9).

    Parameters
    ----------
    probabilities : array-like of shape (10,)
        Softmax probabilities for digits 0 through 9.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object that can be passed directly to st.pyplot().
    """
    probabilities = np.asarray(probabilities, dtype="float32")
    if probabilities.shape != (10,):
        raise ValueError(
            "plot_probability_chart expects 10 class probabilities."
        )

    digits = list(range(10))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(digits, probabilities, color="#4C72B0")

    # Highlight the highest bar (the predicted digit) in a different color
    best_index = int(np.argmax(probabilities))
    bars[best_index].set_color("#DD5555")

    ax.set_xticks(digits)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Probability Distribution (0-9)")

    # Add percentage labels above each bar
    for i, prob in enumerate(probabilities):
        ax.text(i, prob + 0.02, f"{prob*100:.1f}%", ha="center", fontsize=8)

    fig.tight_layout()
    return fig
