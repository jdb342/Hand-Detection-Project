from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import time
import numpy as np
import csv

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
<button onclick="fetch('/capture')">Catpure Image</button>
</body>
</html>
"""

r_capture = False

@app.route('/capture')
def capture():
	global r_capture
	r_capture = True
	return "Capture requested"

def normalize_landmarks(coords,label):
	# Takes list of landmarks: list of (x, y) tuples, length 21
	# and outputs normalized landmarks in array
	
	# Set origin as wrist
	x_wrist, y_wrist = coords[0]
		
	translated = []
	
	# Creates vectors from wrist to landmarks
	for x, y in coords:
		translated.append([x - x_wrist, y - y_wrist])
	
	if label == "Left":	
		for lm in translated:
			lm[0] *= -1
		
	# Scales from wrist to middle finger, smaller the distance between 
	# the two the bigger the hand becomes normalized, vice versa
	scale = np.linalg.norm(np.array(translated[12]))
	
	if scale == 0:
		scale = 1
	
	normalized = []
	
	for x, y in translated:
		normalized.append(x / scale)
		normalized.append(y / scale)
	

		
	return np.array(normalized)

def gen_frames():
	global r_capture
	
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
				
			for hand_landmarks, handedness in zip(
					results.multi_hand_landmarks,
					results.multi_handedness):
						
				label = handedness.classification[0].label
			            
			# Output landmarks into terminal every .5 seconds
			if r_capture:
				
				handLms = results.multi_hand_landmarks[0]
				coords = []
									
				for lm in handLms.landmark:
					coords.append([round(lm.x, 3), round(lm.y, 3)])
					
				normalized_array = normalize_landmarks(coords, label)
				
				print(normalized_array)
				
				# Enter into csv file gesture_data.csv
				with open('/home/jb26/gesture-project-coding/gesture_data.csv', 'a', newline='') as csvfile:
					csv_writer = csv.writer(csvfile)
					csv_writer.writerow(normalized_array)
				
				print(label)
				r_capture = False
				
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
