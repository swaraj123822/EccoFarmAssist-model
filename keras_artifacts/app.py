from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/plant_disease_model.keras")

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    image = preprocess_image(image)
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({"prediction": int(predicted_class)})

if __name__ == "__main__":
    app.run(debug=True)
