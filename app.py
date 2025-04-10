import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, request
import os
import base64
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = app.logger

# Initialize MediaPipe Hands
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.4)
    logger.info("MediaPipe Hands initialized")
except Exception as e:
    logger.error(f"Error initializing MediaPipe: {e}")
    raise

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="model_with_select_tf_ops.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite model loaded successfully")
except Exception as e:
    logger.error(f"Error loading TFLite model: {e}")
    raise

ACTIONS = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
threshold = 0.8

def process_frame(frame):
    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            input_data = np.expand_dims([landmarks], axis=0).astype(np.float32)  # Single frame
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            res = interpreter.get_tensor(output_details[0]['index'])[0]
            prediction = ACTIONS[np.argmax(res)]
            confidence = np.max(res)
            if confidence > threshold:
                return f"{prediction} ({confidence:.2f})"
        return "No hand detected"
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")
        return f"Server error: {str(e)}"

@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received request to /predict")
    try:
        data = request.form['frame']
        frame_data = base64.b64decode(data)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to decode frame")
            return "Error: Invalid frame data"
        result = process_frame(frame)
        logger.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return f"Server error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)