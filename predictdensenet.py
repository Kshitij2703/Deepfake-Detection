import numpy as np
import cv2
from tensorflow.keras.models import load_model
import face_recognition
import matplotlib.pyplot as plt

# Function to display multiple frames in a grid
def display_all_frames(frames, title):
    num_frames = len(frames)
    cols = 5  # Number of columns in the grid
    rows = (num_frames // cols) + (num_frames % cols > 0)  # Calculate rows needed

    plt.figure(figsize=(15, rows * 3))  # Adjust size based on number of frames
    plt.suptitle(title, fontsize=16)

    for idx, frame in enumerate(frames):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.show()

# Function to load and preprocess video with face detection
def load_video(video_path, n_frames=5, size=(112, 112)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    face_detected = False  # Track if any face is detected

    # Iterate over every frame in the video
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames are available

        # Resize the frame to 50% for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face using face_recognition
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            # Scale back the face location to the original frame size
            top, right, bottom, left = [int(coord * 2) for coord in face_locations[0]]
            face_frame = frame[top:bottom, left:right]
            face_detected = True

            # Resize and normalize the face frame
            face_frame = cv2.resize(face_frame, size)
            face_frame = face_frame.astype(np.float32) / 255.0  # Normalize
            frames.append(face_frame)

        # Stop if we have collected the required number of frames
        if len(frames) == n_frames:
            break

    cap.release()

    # If no faces were detected in enough frames, return None
    if not face_detected or len(frames) < n_frames:
        return None

    return np.array(frames)  # Return collected frames


# Load the saved model
model = load_model('D:/Deepfakedata/deepfake_detector_best_densenet121.h5')

# Function to make a prediction on a single video
def predict_video(video_path):
    # Load and preprocess the video
    preprocessed_video = load_video(video_path)
    
    if preprocessed_video is None:
        return "No face detected, cannot classify", None  # Skip prediction if not enough valid frames

    # Display the processed frame
    display_all_frames(preprocessed_video, title="All Processed Frames")

    # Ensure the video data is in the correct shape (1, N_FRAMES, 112, 112, 3)
    video_input = np.expand_dims(preprocessed_video, axis=0)

    # Predict using the model
    prediction = model.predict(video_input)

    # Since it's binary classification, prediction will be a probability
    predicted_label = 'FAKE' if prediction[0] > 0.5 else 'REAL'

    return predicted_label, prediction[0]

# Make a prediction on a video file
predicted_label, probability = predict_video('D:/Deepfakedata/Face_Datasetnew/FAKE/pwftvlkjqp.mp4')
print(f"Predicted Label: {predicted_label}")
print(f"Prediction Probability: {probability}")
