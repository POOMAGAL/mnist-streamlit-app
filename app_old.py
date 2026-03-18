import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🧠 MNIST Digit Classifier")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload a digit image (28x28)", type=["png", "jpg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28, 28))

    st.image(img, caption="Input Image", width=150)

    # Preprocess
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Prediction
    predictions = model(img_array)
    probs = tf.nn.softmax(predictions).numpy()

    predicted_class = np.argmax(probs)

    st.success(f"Predicted Digit: {predicted_class}")

    st.write("### Probabilities:")
    st.bar_chart(probs[0])