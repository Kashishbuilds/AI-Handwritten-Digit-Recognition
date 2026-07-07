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
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. IMAGE PREPROCESSING
# ----------------------------------------------------------------------
def center_digit(gray_image):
    """
    Crop the image tightly around the drawn/written digit, then paste it
    centered onto a padded square canvas.

    Why this matters: real MNIST digits are always centered and scaled to
    fit within a consistent bounding box inside the 28x28 frame. If a user
    draws a small digit off to one side of a canvas (or an uploaded photo
    has lots of empty margin), naively resizing the whole image to 28x28
    shrinks the actual digit down and shifts it off-center. CNNs trained
    on MNIST are quite sensitive to this mismatch, which is a common cause
    of wrong predictions - especially for shape-asymmetric digits like
    1, 7, and 9.

    Parameters
    ----------
    gray_image : np.ndarray
        A single-channel (grayscale) image where the digit is bright
        (white-ish) and the background is dark (black-ish).

    Returns
    -------
    np.ndarray
        A square grayscale image with the digit centered and padded,
        ready to be resized down to 28x28.
    """
    # Find all "on" pixels (part of the digit stroke)
    coords = cv2.findNonZero(gray_image)

    # If the canvas is blank (nothing drawn yet), just return it as-is
    if coords is None:
        return gray_image

    # Get the tight bounding box around the digit
    x, y, w, h = cv2.boundingRect(coords)
    digit = gray_image[y:y + h, x:x + w]

    # Paste the cropped digit onto a square black canvas, centered
    size = max(w, h)
    padded = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = digit

    # Add a margin border (~20% of size) so the digit doesn't touch the
    # edges after resizing - MNIST digits typically have some breathing
    # room around them, not filling the entire 28x28 frame.
    margin = max(int(size * 0.2), 2)
    bordered = cv2.copyMakeBorder(
        padded, margin, margin, margin, margin,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )
    return bordered


def preprocess_image(image):
    """
    Convert any input image (PIL Image, numpy array, or canvas RGBA array)
    into a clean 28x28 grayscale, normalized image ready for the CNN.

    Steps performed:
        1. Convert to a numpy array (if it isn't already one)
        2. Convert to grayscale
        3. Invert colors if needed so the digit is white-on-black
           (MNIST digits are white strokes on a BLACK background)
        4. Crop tightly around the digit and re-center it on a padded
           square canvas (matches how MNIST digits are framed)
        5. Resize to 28x28 (the size MNIST digits use)
        6. Normalize pixel values to the range [0, 1]

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
            image = np.array(image)

        # If the canvas gives us an RGBA image, drop the alpha channel
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Step 2: Convert to grayscale (if not already single channel)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Step 3: Decide whether we need to invert colors *before* cropping,
        # so that center_digit() can correctly find the digit's pixels.
        # MNIST digits are WHITE strokes on a BLACK background.
        # Most uploaded photos / scanned notes are the opposite
        # (dark ink on a light/white background), so we invert them.
        mean_pixel_value = np.mean(gray)
        if mean_pixel_value > 127:
            # Background is bright -> invert so digit becomes white on black
            gray = cv2.bitwise_not(gray)

        # Step 4: Crop tightly around the digit and re-center it, matching
        # how real MNIST digits are framed within the 28x28 box.
        centered = center_digit(gray)

        # Step 5: Resize to 28x28 pixels (MNIST's native size)
        resized = cv2.resize(centered, (28, 28), interpolation=cv2.INTER_AREA)

        # Step 6: Normalize pixel values between 0 and 1
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
        return processed_image.reshape(1, 28, 28, 1)
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
