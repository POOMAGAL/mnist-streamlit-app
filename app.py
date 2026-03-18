import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="wide")

st.title("🧠 Handwritten Digit Recognizer")

# ---------------------------
# Build model + load weights
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.load_weights("weights.h5")
    return model

model = load_model()

# ---------------------------
# Layout: Side by Side
# ---------------------------
col1, col2 = st.columns(2)

# ---------------------------
# ✍️ Canvas Input
# ---------------------------
with col1:
    st.subheader("✍️ Draw Digit")

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

    if st.button("Predict from Canvas"):
        if canvas_result.image_data is not None:

            img = canvas_result.image_data
            img = Image.fromarray((img[:, :, 0]).astype('uint8'))
            img = img.resize((28, 28))

            st.image(img, caption="Processed Canvas", width=150)

            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28)

            predictions = model(img_array)
            probs = tf.nn.softmax(predictions).numpy()

            st.success(f"Prediction: {np.argmax(probs)}")
            st.bar_chart(probs[0])


# ---------------------------
# 📤 Upload Image
# ---------------------------
with col2:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        img = img.resize((28, 28))

        st.image(img, caption="Uploaded Image", width=150)

        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        predictions = model(img_array)
        probs = tf.nn.softmax(predictions).numpy()

        st.success(f"Prediction: {np.argmax(probs)}")
        st.bar_chart(probs[0])