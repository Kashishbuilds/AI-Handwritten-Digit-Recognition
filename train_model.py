"""
train_model.py
---------------
This script trains a Convolutional Neural Network (CNN) on the MNIST
handwritten digit dataset and saves the trained model as `model.h5`.

Run this file once before running the Streamlit app:
    python train_model.py

It will:
    1. Download / load the MNIST dataset from Keras
    2. Preprocess the data (normalize + reshape)
    3. Build a CNN using Conv2D, MaxPooling2D, Dropout, and Dense layers
    4. Train the model for ~10 epochs
    5. Evaluate it on the test set
    6. Plot training vs validation accuracy/loss graphs
    7. Save the trained model to model.h5
"""

import os
import matplotlib.pyplot as plt


def get_tensorflow_components():
    """Lazy import TensorFlow to avoid top-level import failures."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.datasets import mnist
        return tf, layers, models, mnist
    except ImportError as error:
        raise ImportError(
            "TensorFlow could not be imported. "
            "This usually means the installed TensorFlow package is not "
            "compatible with your CPU or operating system. "
            "Try installing a compatible build, for example: \n"
            "    pip install --upgrade pip\n"
            "    pip install tensorflow-cpu\n"
            "or use a slightly older TensorFlow version if your CPU lacks AVX/AVX2 support.\n"
            "Also make sure the Microsoft Visual C++ Redistributable is installed.\n"
            f"Original import error: {error}"
        ) from error


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 128
MODEL_SAVE_PATH = "model.h51"
PLOT_SAVE_PATH = os.path.join("assets", "screenshots", "training_history.png")


def load_data():
    """
    Load the MNIST dataset and preprocess it:
        - Normalize pixel values to [0, 1]
        - Reshape images to (28, 28, 1) since CNNs expect a channel dimension
    """
    print("Loading MNIST dataset...")
    _, _, _, mnist = get_tensorflow_components()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape from (N, 28, 28) to (N, 28, 28, 1) - CNN needs a channel axis
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """
    Build and compile the CNN architecture.

    Architecture:
        Conv2D(32) -> MaxPooling2D -> Conv2D(64) -> MaxPooling2D
        -> Dropout -> Flatten -> Dense(128, relu) -> Dropout
        -> Dense(10, softmax)
    """
    _, layers, models, _ = get_tensorflow_components()
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
                      input_shape=(28, 28, 1), padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Regularization to prevent overfitting
        layers.Dropout(0.25),

        # Flatten feature maps into a 1D vector for the Dense layers
        layers.Flatten(),

        # Fully connected (Dense) layer
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        # Output layer: 10 classes (digits 0-9), softmax gives probabilities
        layers.Dense(10, activation="softmax"),
    ])

    # Adam optimizer + sparse categorical crossentropy since labels are
    # plain integers (0-9), not one-hot encoded
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def plot_training_history(history):
    """
    Plot training vs validation accuracy and loss, and save the figure
    to the assets/screenshots folder.
    """
    os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy plot
    ax1.plot(history.history["accuracy"], label="Training Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss plot
    ax2.plot(history.history["loss"], label="Training Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH)
    print(f"Training history graph saved to: {PLOT_SAVE_PATH}")
    plt.close(fig)


def main():
    try:
        # 1. Load and preprocess the data
        (x_train, y_train), (x_test, y_test) = load_data()

        # 2. Build the CNN
        model = build_model()

        # 3. Train the model, holding out part of the training set for validation
        print(f"\nStarting training for {EPOCHS} epochs...\n")
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.1,   # use 10% of training data for validation
            verbose=1,
        )

        # 4. Evaluate on the untouched test set
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("\n===================================")
        print(f"Final Test Accuracy : {test_accuracy * 100:.2f}%")
        print(f"Final Test Loss     : {test_loss:.4f}")
        print("===================================\n")

        # 5. Plot and save training/validation curves
        plot_training_history(history)

        # 6. Save the trained model
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved successfully to: {MODEL_SAVE_PATH}")

    except Exception as error:
        print(f"An error occurred during training: {error}")
        raise


if __name__ == "__main__":
    main()
