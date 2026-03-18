import tensorflow as tf

# Load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()

# Normalize
x_train = x_train / 255.0

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5)

# Save model
model.save("model.keras")