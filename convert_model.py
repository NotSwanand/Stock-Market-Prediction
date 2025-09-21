from tensorflow.keras.models import load_model

# Load old .h5 model
model = load_model("models/lstm_model.h5")

# Save in new Keras format
model.save("models/lstm_model.keras")
