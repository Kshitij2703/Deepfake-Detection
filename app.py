from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import time
import face_recognition
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FRAME_FOLDER'] = 'static/frames'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

MODEL_PATH = r"C:\Users\Kshitij navale\OneDrive\Desktop\gui\deepfake_detector_best_densenet121.keras"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}
SEQUENCE_LENGTH = 5
FRAME_HEIGHT = 112
FRAME_WIDTH = 112

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAME_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_deepfake_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model = load_deepfake_model()

def clear_previous_frames():
    for filename in os.listdir(app.config['FRAME_FOLDER']):
        file_path = os.path.join(app.config['FRAME_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {str(e)}")

def load_video_with_face_detection(video_path):
    frames = []
    timestamp = int(time.time())
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            # Scale back the face location to original frame size
            top, right, bottom, left = [int(coord * 2) for coord in face_locations[0]]
            face_frame = frame[top:bottom, left:right]
            
            # Resize face frame to required dimensions
            face_frame = cv2.resize(face_frame, (FRAME_WIDTH, FRAME_HEIGHT))
            normalized_frame = face_frame.astype(np.float32) / 255.0
            
            frames.append(normalized_frame)
            
            # Save the processed frame
            if frame_count < SEQUENCE_LENGTH:
                frame_filename = os.path.join(
                    app.config['FRAME_FOLDER'],
                    f'frame_{frame_count}_{timestamp}.jpg'
                )
                cv2.imwrite(frame_filename, face_frame)
                frame_paths.append(frame_filename)
                frame_count += 1

        if frame_count >= SEQUENCE_LENGTH:
            break

    cap.release()

    if len(frames) < SEQUENCE_LENGTH:
        return None, None, "Not enough face frames detected"

    return np.array(frames), timestamp, None

def process_video(video_path: str):
    # Get frames with face detection
    frames, timestamp, error = load_video_with_face_detection(video_path)
    
    if error:
        return {"error": error}
    
    if frames is None or len(frames) < SEQUENCE_LENGTH:
        return {"error": "Failed to process video - insufficient face frames detected"}

    # Prepare frames for prediction
    preprocessed_frames = np.expand_dims(frames[:SEQUENCE_LENGTH], axis=0)
    
    # Make prediction
    prediction = model.predict(preprocessed_frames)[0][0]
    
    # Calculate confidence
    is_fake = bool(prediction >= 0.5)
    confidence = prediction * 100 if is_fake else (1 - prediction) * 100
    
    # Generate frame URLs
    frame_urls = [
        url_for('static', filename=f'frames/frame_{i}_{timestamp}.jpg')
        for i in range(SEQUENCE_LENGTH)
    ]
    
    result = {
        "is_fake": is_fake,
        "frame_urls": frame_urls,
        "timestamp": timestamp
    }
    
    # Add only the relevant confidence
    if is_fake:
        result["fake_confidence"] = confidence
    else:
        result["real_confidence"] = confidence
    
    return result

@app.route('/')
def index():
    return render_template('interface.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No video file selected"})
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"})
    
    try:
        # Clear previous frames
        clear_previous_frames()

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = process_video(filepath)
        
        if "error" in result:
            return jsonify(result)
            
        result["video_url"] = url_for('static', filename=f'uploads/{filename}')
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)})

# Add headers to prevent caching
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(debug=True)