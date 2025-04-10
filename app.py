import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, Response, render_template, request
import os
import base64
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)  # Enable logging

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.4)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_with_select_tf_ops.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define actions and parameters
ACTIONS = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
SEQUENCE_LENGTH = 30
threshold = 0.8
sequence = []

def process_frame(frame):
    global sequence
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        sequence.append(landmarks)
        
        if len(sequence) > SEQUENCE_LENGTH:
            sequence = sequence[-SEQUENCE_LENGTH:]
        
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            res = interpreter.get_tensor(output_details[0]['index'])[0]
            prediction = ACTIONS[np.argmax(res)]
            confidence = np.max(res)
            if confidence > threshold:
                return f"{prediction} ({confidence:.2f})"
    return "No hand detected"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Received request to /predict")  # Log request
    data = request.form['frame']
    frame_data = base64.b64decode(data)
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = process_frame(frame)
    app.logger.info(f"Prediction result: {result}")  # Log result
    return result

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)