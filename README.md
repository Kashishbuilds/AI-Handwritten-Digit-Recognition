# 🔢 AI Handwritten Digit Recognition System

A complete, 100% Python-based deep learning application that recognizes
handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained
on the MNIST dataset. The entire user interface is built with **Streamlit**
— no HTML, CSS, or JavaScript files required.

Users can either **draw a digit** on an interactive canvas or **upload an
image** of a handwritten digit, and the app will predict the digit along
with a confidence score and a full probability breakdown.

---

## 📌 Project Overview

Handwritten digit recognition is one of the classic "Hello World" problems
of computer vision and deep learning. This project takes it a step further
by wrapping a trained CNN inside a fully interactive, easy-to-use Streamlit
web application — all written in Python.

The project is split into clear, reusable modules:

| File               | Purpose                                                          |
|--------------------|-------------------------------------------------------------------|
| `train_model.py`   | Loads MNIST, builds & trains the CNN, saves `model.h5`            |
| `predict.py`       | Loads the trained model and runs predictions                     |
| `utils.py`         | Shared image preprocessing and chart-plotting helper functions   |
| `app.py`           | The Streamlit web application (UI)                               |
| `model.h5`         | The saved, pre-trained CNN model                                  |
| `requirements.txt` | All Python dependencies needed to run the project                |

---

## ✨ Features

- ✏️ **Draw digits** directly in the browser using an interactive canvas
- 📁 **Upload images** of handwritten digits (PNG/JPG/JPEG)
- 🧹 Automatic image preprocessing:
  - Grayscale conversion
  - Resizing to 28×28 (MNIST's native resolution)
  - Pixel normalization (0–1 range)
  - Automatic color inversion (white digit on black background)
- 🧠 CNN-based digit prediction (digits 0–9)
- 📊 Confidence score with a live progress bar
- 📈 Full probability distribution bar chart across all 10 digit classes
- 🗂️ Session-based **prediction history** sidebar
- 📉 Training vs validation accuracy/loss graphs generated during training
- 🛡️ Proper exception handling throughout (missing model, bad images, etc.)
- 🎨 Clean, modern Streamlit UI with sidebar navigation

---

## 🛠️ Technologies Used

- **Python 3** — core programming language
- **Streamlit** — 100% Python web UI framework
- **TensorFlow / Keras** — deep learning framework used to build and train the CNN
- **NumPy** — numerical array operations
- **OpenCV (`opencv-python-headless`)** — image processing (grayscale, resize, invert)
- **Pillow (PIL)** — image loading and format handling
- **Matplotlib** — accuracy/loss graphs and probability distribution charts
- **streamlit-drawable-canvas** — interactive drawing canvas widget for Streamlit

---

## 📚 Dataset Information: MNIST

The **MNIST (Modified National Institute of Standards and Technology)**
dataset is the standard benchmark dataset for handwritten digit recognition.

- 70,000 total grayscale images of handwritten digits (0–9)
  - 60,000 training images
  - 10,000 test images
- Each image is **28×28 pixels**, grayscale (single channel)
- Digits appear as **white strokes on a black background**
- Loaded directly via `tensorflow.keras.datasets.mnist`, so no manual
  download is required — Keras fetches it automatically the first time
  `train_model.py` is run.

---

## 🧠 CNN Architecture Explanation

The model is a **Convolutional Neural Network (CNN)**, which is far better
suited to image data than a plain Artificial Neural Network (ANN) because
it preserves spatial relationships between pixels.

```
Input (28x28x1)
    │
Conv2D (32 filters, 3x3, ReLU)      -> learns simple edges/curves
    │
MaxPooling2D (2x2)                   -> downsamples, reduces computation
    │
Conv2D (64 filters, 3x3, ReLU)      -> learns more complex shapes
    │
MaxPooling2D (2x2)                   -> downsamples further
    │
Dropout (0.25)                       -> reduces overfitting
    │
Flatten                              -> converts 2D feature maps to 1D vector
    │
Dense (128 units, ReLU)              -> learns high-level combinations
    │
Dropout (0.5)                        -> further regularization
    │
Dense (10 units, Softmax)            -> outputs probability for each digit (0-9)
```

**Key design choices:**
- **Conv2D + MaxPooling2D** layers extract and compress visual features
  (edges, curves, loops) that make up each digit.
- **Dropout** layers randomly disable neurons during training to prevent
  the model from memorizing the training data (overfitting).
- **ReLU activation** introduces non-linearity, allowing the network to
  learn complex patterns.
- **Softmax output layer** converts the final layer's raw scores into a
  probability distribution across the 10 digit classes.
- **Adam optimizer** — an efficient, adaptive learning-rate optimizer.
- **Sparse categorical crossentropy** — appropriate loss function since our
  labels are plain integers (0–9) rather than one-hot encoded vectors.

---

## ⚙️ Installation Steps

1. **Clone or download this project folder.**

2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Run Commands

1. **Train the model** (only needs to be done once; downloads MNIST
   automatically and saves `model.h5`):
   ```bash
   python train_model.py
   ```

2. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. Open the local URL shown in your terminal (usually
   `http://localhost:8501`) in your browser.

> 💡 You only need to run `train_model.py` once. After that, `app.py` will
> simply load the saved `model.h5` file for instant predictions.

---

## 📁 Folder Structure

```
AI-Handwritten-Digit-Recognition/
│
├── app.py                     # Streamlit UI application
├── train_model.py             # CNN training script
├── predict.py                 # Model loading & prediction logic
├── utils.py                   # Shared preprocessing / plotting helpers
├── requirements.txt           # Python dependencies
├── model.h5                   # Saved trained CNN model
├── README.md                  # Project documentation (this file)
├── interview_questions.md     # Interview prep & project explanation
└── assets/
    └── screenshots/
        └── training_history.png   # Training/validation accuracy & loss graphs
```

---

## 🚀 Future Enhancements

- Support recognizing multi-digit sequences (e.g., full numbers, not just single digits)
- Add support for other datasets (EMNIST for letters, custom handwriting datasets)
- Deploy the app publicly via Streamlit Community Cloud or Hugging Face Spaces
- Add a model comparison mode (CNN vs ANN vs SVM) with side-by-side accuracy
- Add data augmentation (rotation, shifting, zoom) during training to
  improve robustness against messy real-world handwriting
- Add a REST API (FastAPI) layer so the model can be used outside Streamlit
- Add user authentication and persistent (database-backed) prediction history
- Convert the model to TensorFlow Lite for mobile/edge deployment

---

## 📝 License

This project is provided for educational and portfolio purposes.
