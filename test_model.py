import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pickle

# Load model
model = tf.keras.models.load_model("models/best_model.keras")

# Load class names
with open("models/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# Test dataset path
TEST_PATH = "Datasets/train"

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
loss, accuracy = model.evaluate(test_data)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predictions
predictions = model.predict(test_data, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)