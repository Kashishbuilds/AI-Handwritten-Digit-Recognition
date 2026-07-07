"""
app.py
------
Main Streamlit application for the AI Handwritten Digit Recognition System.

Features:
    - Draw a digit on a canvas OR upload an image of a handwritten digit
    - Preprocess the image (grayscale, resize, normalize, invert)
    - Predict the digit using a pre-trained CNN (model.h5)
    - Show predicted digit, confidence progress bar, and probability chart
    - Keep a running history of predictions made in the session

Run with:
    streamlit run app.py
"""

import os
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from predict import load_trained_model, predict_digit
from utils import plot_probability_chart

# ----------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model.h51"


# ----------------------------------------------------------------------
# CACHED MODEL LOADING
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained CNN model...")
def get_model():
    """
    Load the trained model once and cache it across Streamlit reruns,
    so we don't reload it from disk on every interaction.
    """
    return load_trained_model(MODEL_PATH)


# ----------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []          # list of past predictions

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0        # used to force-clear the canvas


def add_to_history(source, digit, confidence):
    """Append a new prediction record to the session history."""
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "digit": digit,
        "confidence": confidence,
    })
    # Keep only the last 10 predictions to avoid clutter
    st.session_state.history = st.session_state.history[:10]


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🔢 Digit Recognizer")
    st.markdown("Use the options below to configure the app.")

    st.divider()
    st.subheader("Input Mode")
    input_mode = st.radio(
        "Choose how you want to provide a digit:",
        ("✏️ Draw a Digit", "📁 Upload an Image"),
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("About")
    st.info(
        "This app uses a Convolutional Neural Network (CNN) trained on the "
        "**MNIST** dataset to recognize handwritten digits (0-9).\n\n"
        "Built with **Streamlit**, **TensorFlow/Keras**, **OpenCV**, and **Pillow**."
    )

    st.divider()
    st.subheader("🕘 Prediction History")
    if st.session_state.history:
        for record in st.session_state.history:
            st.write(
                f"`{record['time']}` — **{record['digit']}** "
                f"({record['confidence']:.1f}%) via {record['source']}"
            )
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions made yet.")


# ----------------------------------------------------------------------
# MAIN TITLE SECTION
# ----------------------------------------------------------------------
st.title("🔢 AI Handwritten Digit Recognition System")
st.markdown(
    "Draw a digit or upload a handwritten digit image, and let the "
    "AI model predict which digit (0-9) it is, along with a confidence score."
)
st.divider()


# ----------------------------------------------------------------------
# LOAD MODEL (with error handling in case model.h51 is missing)
# ----------------------------------------------------------------------
model = None
model_error = None
try:
    model = get_model()
except FileNotFoundError as fnf_error:
    model_error = str(fnf_error)
except Exception as generic_error:
    model_error = f"Unexpected error while loading the model: {generic_error}"

if model_error:
    st.error(model_error)
    st.warning(
        "👉 Please run `python train_model.py` first in your terminal to "
        "train and save the model before using this app."
    )
    st.stop()


# ----------------------------------------------------------------------
# MAIN LAYOUT: two columns -> input on the left, results on the right
# ----------------------------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

input_image = None       # will hold the PIL/np image to predict on
source_label = ""         # "Canvas" or "Upload", used for history logging

# ------------------------- LEFT COLUMN: INPUT -------------------------
with left_col:
    if input_mode == "✏️ Draw a Digit":
        st.subheader("Draw a digit below")
        st.caption("Use your mouse or touchscreen to draw a single digit (0-9).")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            predict_clicked = st.button("🔍 Predict Digit", use_container_width=True)
        with button_col2:
            if st.button("🧹 Clear Canvas", use_container_width=True):
                st.session_state.canvas_key += 1  # forces a fresh canvas widget
                st.rerun()

        if canvas_result.image_data is not None:
            input_image = canvas_result.image_data.astype("uint8")
            source_label = "Canvas"

    else:  # Upload mode
        st.subheader("Upload a handwritten digit image")
        st.caption("Supported formats: PNG, JPG, JPEG")

        uploaded_file = st.file_uploader(
            "Choose an image file", type=["png", "jpg", "jpeg"]
        )

        predict_clicked = st.button("🔍 Predict Digit", use_container_width=True)

        if uploaded_file is not None:
            try:
                input_image = Image.open(uploaded_file).convert("RGB")
                source_label = "Upload"
                st.image(input_image, caption="Uploaded Image", width=200)
            except Exception as upload_error:
                st.error(f"Could not read the uploaded image: {upload_error}")
                input_image = None


# ------------------------- RIGHT COLUMN: RESULTS -----------------------
with right_col:
    st.subheader("Prediction Results")

    if predict_clicked:
        if input_image is None:
            st.warning("⚠️ Please draw a digit or upload an image first.")
        else:
            try:
                with st.spinner("Running prediction..."):
                    result = predict_digit(model, input_image)

                digit = result["digit"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]

                # Big, clear predicted digit display
                st.markdown(
                    f"<h1 style='text-align:center; font-size:90px; "
                    f"color:#DD5555;'>{digit}</h1>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='text-align:center; font-size:20px;'>"
                    f"Predicted Digit</p>",
                    unsafe_allow_html=True,
                )

                # Confidence progress bar
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.progress(min(int(confidence), 100))

                # Preview of what the model actually "saw" after preprocessing
                with st.expander("🔬 View Preprocessed 28x28 Image"):
                    st.image(
                        result["processed_image"],
                        caption="Preprocessed Image (fed into the CNN)",
                        width=140,
                        clamp=True,
                    )

                # Probability distribution chart
                st.write("**Probability Distribution (0-9):**")
                fig = plot_probability_chart(probabilities)
                st.pyplot(fig)

                # Log this prediction into history
                add_to_history(source_label, digit, confidence)

            except Exception as prediction_error:
                st.error(f"Prediction failed: {prediction_error}")
    else:
        st.info("Draw or upload a digit, then click **Predict Digit** to see results here.")


st.divider()
st.caption(
    "Built with ❤️ using Python, Streamlit, TensorFlow/Keras, OpenCV, and Pillow. "
    "Model trained on the MNIST dataset."
)
