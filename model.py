import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GlobalAveragePooling2D

# Load Pretrained ResNet50 (Feature Extractor)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define Image Model (ResNet50 + Dense)
image_model = Sequential([
    resnet_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary Classification (Real/Fake)
])
image_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Video Model (LSTM for Frame Sequences)
video_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(10, 128)),  # 10 frames, 128 features
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
video_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to Preprocess Image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to Process Video (Extract Frames)
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 10:  # Extract 10 frames
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (224, 224)) / 255.0  # Normalize
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    
    # Pad if fewer than 10 frames
    if len(frames) < 10:
        frames = np.pad(frames, ((0, 10 - len(frames)), (0, 0), (0, 0), (0, 0)), mode='constant')

    features = resnet_model.predict(frames)
    return np.mean(features, axis=0)  # Average over frames

# Prediction Function
def predict_file(filepath):
    if filepath.endswith(('.jpg', '.png', '.jpeg')):  # Image Input
        img_array = preprocess_image(filepath)
        prediction = image_model.predict(img_array)[0][0]
    elif filepath.endswith(('.mp4', '.avi', '.mov')):  # Video Input
        video_features = preprocess_video(filepath)
        video_features = np.expand_dims(video_features, axis=0)  # Expand dims for LSTM
        prediction = video_model.predict(video_features)[0][0]
    else:
        return "Unsupported file format"

    return "Fake" if prediction > 0.5 else "Real"
