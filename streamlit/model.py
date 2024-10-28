import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
import atexit
from typing import List

# Configuration
MODEL_PATH = "C:/Users/Kshitij navale/OneDrive/Desktop/gui/deepfake_detector_best_densenet121.keras"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "webm"}
SEQUENCE_LENGTH = 5
FRAME_HEIGHT = 112
FRAME_WIDTH = 112

# Global variable to store temporary files
temp_files = []

def cleanup_temp_files():
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass

atexit.register(cleanup_temp_files)

@st.cache_resource
def load_deepfake_model():
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_frames_from_video(video_path: str) -> List[np.ndarray]:
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    
    return frames

def preprocess_frames(frames: List[np.ndarray]) -> np.ndarray:
    processed_frames = []
    
    for frame in frames:
        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed_frames.append(rgb_frame)
    
    processed_frames = np.array(processed_frames, dtype='float32') / 255.0
    return processed_frames

def create_sequences(frames: np.ndarray, sequence_length: int) -> List[np.ndarray]:
    sequences = []
    for i in range(0, len(frames) - sequence_length + 1, sequence_length):
        sequence = frames[i:i + sequence_length]
        if len(sequence) == sequence_length:
            sequences.append(sequence)
    return sequences

def process_video(video_path: str, model) -> float:
    if model is None:
        st.error("Model not loaded properly.")
        return 0
    
    try:
        frames = get_frames_from_video(video_path)
        if not frames:
            st.error("No frames could be extracted from the video.")
            return 0
        
        processed_frames = preprocess_frames(frames)
        sequences = create_sequences(processed_frames, SEQUENCE_LENGTH)
        
        if not sequences:
            st.error("Could not create enough sequences from the video.")
            return 0
        
        predictions = []
        for sequence in sequences:
            sequence_batch = np.expand_dims(sequence, axis=0)
            with tf.device('/CPU:0'):
                prediction = model.predict(sequence_batch, verbose=0)[0][0]
            predictions.append(prediction)
        
        final_prediction = np.median(predictions) * 100
        authenticity_score = 100 - final_prediction
        return authenticity_score
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return 0

# Initialize model
model = load_deepfake_model()

# Streamlit app
st.title("Deepfake Video Detector")
st.write("Upload a video file, and this tool will predict if it's real or fake.")

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

uploaded_file = st.file_uploader("Choose a video file...", type=list(ALLOWED_EXTENSIONS))

if uploaded_file is not None:
    try:
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Add to cleanup list
        temp_files.append(tfile.name)
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display video using Streamlit's native video player
            video_file = open(tfile.name, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_file.close()
        
        # Process video
        if st.button("Analyze Video") or st.session_state.analyzed:
            if model is None:
                st.error("Model is not loaded.")
            else:
                st.session_state.analyzed = True
                with st.spinner("Processing video..."):
                    confidence = process_video(tfile.name, model)
                    is_fake = confidence < 50
                    
                    # Show result
                    result_container = st.container()
                    with result_container:
                        result_text = "FAKE" if is_fake else "REAL"
                        st.markdown(
                            f"""
                            <div style='padding: 20px; 
                                      border-radius: 10px; 
                                      background-color: {"rgba(255,0,0,0.1)" if is_fake else "rgba(0,255,0,0.1)"}; 
                                      margin-bottom: 20px;'>
                                <h2 style='margin: 0; 
                                         color: {"red" if is_fake else "green"}; 
                                         text-align: center;'>
                                    {result_text}
                                </h2>
                                <p style='margin: 10px 0 0 0; 
                                        text-align: center; 
                                        font-size: 1.2em;'>
                                    Confidence: {confidence:.1f}%
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Display confidence bar
                        conf_value = min(max(confidence/100, 0.0), 1.0)
                        st.progress(conf_value if not is_fake else 1.0 - conf_value)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Sidebar information
if model is None:
    st.sidebar.error("⚠️ Model not loaded")
else:
    st.sidebar.success("✅ Model loaded successfully")

st.sidebar.info("Note: Temporary files will be cleaned up when you close the application.")