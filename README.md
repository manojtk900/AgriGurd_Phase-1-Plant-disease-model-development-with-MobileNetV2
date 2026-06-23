🌱 AgriGuard AI
AI-Based Smart Agriculture Decision Support System with Plant Disease Detection
👨‍💻 Developed By

Manoj T K
Department of Computer Science and Engineering

📌 Project Overview

AgriGuard AI is a final-year AI/ML-based Smart Agriculture platform designed to assist farmers in disease detection, crop health monitoring, weather-aware decision making, and agricultural recommendations.

The project combines:

Deep Learning
Computer Vision
Explainable AI
Agriculture Analytics
Decision Support Systems

to provide intelligent farming assistance.

🚀 Current Progress
✅ Phase 1 Complete
Plant Disease Detection Model Development

This phase focuses on building a high-accuracy deep learning model capable of identifying plant diseases from leaf images.

Achievements
Dataset Size: 70,000+ Images
Number of Classes: 38
Transfer Learning using MobileNetV2
Image Augmentation
Fine-Tuning
Validation Accuracy: 97.52%
TensorFlow / Keras Implementation
🧠 Model Architecture
Backbone Network

MobileNetV2 (Pretrained on ImageNet)

Additional Layers
Global Average Pooling
Batch Normalization
Dense Layer (256 neurons)
Dropout Layer
Softmax Classification Layer
Training Techniques
Transfer Learning
Data Augmentation
Early Stopping
ReduceLROnPlateau
Model Checkpointing




📊 Results
Metric	Value
Validation Accuracy	97.52%
Training Accuracy	95.06%
Classes	38
Dataset Size	70,000+ Images



📂 Project Structure
AgriGuard_AI/

├── app.py
├── test_model.py
├── templates/
├── static/
├── Datasets/
├── newmodels/
├── efficiantcode.ipynb
└── README.md
🌿 Supported Disease Categories

Examples:

Apple Scab
Apple Cedar Rust
Corn Common Rust
Potato Early Blight
Potato Late Blight
Tomato Early Blight
Tomato Yellow Leaf Curl Virus
Healthy Leaves

and many more.

🛠 Technologies Used
Programming Language
Python
Libraries
TensorFlow
Keras
NumPy
Matplotlib
OpenCV
Framework
Flask
