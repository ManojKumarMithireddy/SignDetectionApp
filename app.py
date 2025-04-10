import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, Response, render_template

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.4
)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_with_select_tf_ops.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define actions and parameters
ACTIONS = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
SEQUENCE_LENGTH = 30
sequence = []
predictions = []
threshold = 0.8

# Initialize webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    global sequence, predictions
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Process frame with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            
            # Add to sequence
            sequence.append(landmarks)
            
            # Keep only the last SEQUENCE_LENGTH frames
            if len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[-SEQUENCE_LENGTH:]
            
            # Make prediction when sequence is complete
            if len(sequence) == SEQUENCE_LENGTH:
                # Prepare input for TFLite
                input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                # Run inference
                interpreter.invoke()
                
                # Get output
                res = interpreter.get_tensor(output_details[0]['index'])[0]
                predictions.append(np.argmax(res))
                
                # Smooth predictions (optional)
                if len(predictions) > 5:
                    predictions = predictions[-5:]
                
                # Get final prediction
                current_prediction = ACTIONS[np.argmax(res)]
                confidence = np.max(res)
                
                # Overlay prediction on frame if confident
                if confidence > threshold:
                    cv2.putText(frame, f'{current_prediction.upper()} ({confidence:.2f})', 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use $PORT from env, default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=False)