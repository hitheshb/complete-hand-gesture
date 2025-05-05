# app.py - Flask web application for Indian Sign Language Detection
from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import threading
import time
import os

app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Load the saved model from file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")
model = keras.models.load_model(model_path)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

# Global variables for the current prediction
current_prediction = ""
last_prediction_time = 0
prediction_cooldown = 1.0  # seconds between predictions to reduce processing

# Function to calculate landmark list from hand landmarks
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# Function to preprocess landmark data
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list))) if list(map(abs, temp_landmark_list)) else 1.0

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def generate_frames():
    global current_prediction, last_prediction_time
    
    # For webcam input:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to reduce lag
    
    frame_skip = 2  # Process every nth frame to reduce lag
    frame_count = 0
    
    with mp_hands.Hands(
        model_complexity=0,  # Use lightest model
        max_num_hands=1,     # Only track one hand to improve performance
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5) as hands:

        while True:
            success, image = cap.read()
            frame_count += 1
            
            # Skip frames for performance improvement
            if not success or frame_count % frame_skip != 0:
                continue
                
            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
            
            # Convert to RGB for MediaPipe
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(image)
            
            # Convert back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            current_time = time.time()
            
            # Only perform prediction if enough time has passed since last one
            if results.multi_hand_landmarks and (current_time - last_prediction_time) > prediction_cooldown:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the landmarks on the image
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Extract and preprocess landmarks
                    landmark_list = calc_landmark_list(image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    # Make prediction
                    df = pd.DataFrame(pre_processed_landmark_list).transpose()
                    if not df.empty:
                        predictions = model.predict(df, verbose=0)
                        predicted_class = np.argmax(predictions, axis=1)[0]
                        current_prediction = alphabet[predicted_class]
                        last_prediction_time = current_time
            
            # Add prediction text to the image
            if current_prediction:
                cv2.putText(image, current_prediction, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
            # Add title to the image
            cv2.putText(image, "Indian Sign Language Detection", (120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)
            
            # Convert to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('gesture-to-speech.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_prediction')
def get_current_prediction():
    return jsonify({"prediction": current_prediction})

@app.route('/test-css')
def test_css():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CSS Test</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    </head>
    <body>
        <h1>CSS Test Page</h1>
        <p>If you see styled content, the CSS is working.</p>
        <div style="margin-top: 20px;">
            <a href="/">Back to main page</a>
        </div>
    </body>
    </html>
    """

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5001)