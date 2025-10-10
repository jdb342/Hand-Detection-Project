from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# HTML template for browser
HTML_PAGE = """
<html>
<head>
<title>Hand Detection Stream</title>
</head>
<body>
<h1>Hand Detection Live Stream</h1>
<img src="{{ url_for('video_feed') }}">
</body>
</html>
"""

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw landmarks on the frame
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run Flask on all interfaces so other devices can access it if needed
    app.run(host='0.0.0.0', port=5000)
