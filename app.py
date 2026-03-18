import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("🧠 Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) below 👇")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Convert to image
        img = canvas_result.image_data

        # Convert to grayscale + resize
        img = Image.fromarray((img[:, :, 0]).astype('uint8'))
        img = img.resize((28, 28))

        st.image(img, caption="Processed Image", width=150)

        # Normalize
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        # Predict
        predictions = model(img_array)
        probs = tf.nn.softmax(predictions).numpy()

        predicted_class = np.argmax(probs)

        st.success(f"Predicted Digit: {predicted_class}")
        st.write(f"Confidence: {np.max(probs)*100:.2f}%")

        st.write("### Probabilities")
        st.bar_chart(probs[0])