from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load Model
model = tf.keras.models.load_model("models/best_model.keras")

# Load Class Names
with open("models/class_names.pkl", "rb") as f:
    classes = pickle.load(f)

print("✅ Model Loaded Successfully")
print("✅ Classes Loaded Successfully")
print("Total Classes:", len(classes))
print("Output Shape:", model.output_shape)

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "❌ No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "❌ No file selected"

    # Create static folder if not exists
    os.makedirs("static", exist_ok=True)

    # Secure filename
    filename = secure_filename(file.filename)

    # Save uploaded image
    filepath = os.path.join("static", filename)
    file.save(filepath)

    # Load image
    img = image.load_img(filepath, target_size=(224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]

    confidence = round(float(np.max(prediction)) * 100, 2)

    # Beautify class name
    predicted_class = predicted_class.replace("___", " - ")

    # Image path for HTML
    display_path = "/" + filepath.replace("\\", "/")

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=confidence,
        image_path=display_path
    )

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)