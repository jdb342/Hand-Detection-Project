0.0,0.0,-0.12474027110459297,-0.07878332911869024,-0.23142602928615283,-0.19859964215336526,-0.3348291487544339,-0.283948248698613,-0.43166699079615745,-0.3348291487544339,-0.11489235496475668,-0.4776239327820602,-0.17890380987369264,-0.6565277426557529,-0.215012835719759,-0.7714200976205096,-0.24127394542598912,-0.8731818977321514,-0.019695832279672604,-0.5088090005582085,-0.021337151636311988,-0.730387113704525,-0.024619790349590755,-0.8764645364454302,-0.029543748419508905,-0.9995634881933838,0.06729409362221474,-0.4759826134254208,0.10832707753819933,-0.6778648942920649,0.12966422917451131,-0.8108117621798548,0.141153464670987,-0.932269394571169,0.1395121453143476,-0.3971992843067305,0.21993679378967723,-0.5088090005582085,0.26753505513221937,-0.5974402458167352,0.3036440809782858,-0.682788852361983
from flask import Flask, Response, render_template_string, render_template
import cv2
import mediapipe as mp
import time
import numpy as np
import csv
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
current_gesture = "none"

# Intialize gesture names
gestures = [
"fist",
"open_hand"
]
 
# Initialize K-Means model
X = np.loadtxt("gesture_data.csv", delimiter = ",")
kmeans = KMeans(n_clusters = 2, random_state = 0)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

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
<button onclick="fetch('/capture')">
Catpure Image
</button>
<p>Detected Gesture: <span id="gesture">None</span><p>
<script>
setInterval(() => {
	fetch('/gesture')
		.then(response => response.text())
		.then(data => {
			document.getElementById('gesture').innerText = data;
		});
}, 500); //update every 500 ms
</script>
</body>
</html>
"""

r_capture = False

# To capture moments
@app.route('/capture')
def capture():
	global r_capture
	r_capture = True
	return "Capture requested"
	
# Displays gesture live on web page
@app.route('/gesture')
def gesture(): 
	return current_gesture
	
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
	global current_gesture
	
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
			
			# Maybe this is the distances away from each centroid
			distances = np.linalg.norm(
							X - kmeans.cluster_centers_[kmeans.labels_],
							axis = 1
						)
						
			thresholds = {}
			
			for i in range(kmeans.n_clusters):
				cluster_dists = distances[kmeans.labels_ == i]
				thresholds[i] = cluster_dists.mean() + 1 * cluster_dists.std()
			
			if r_capture:
				
				# Create list of coords, list of [x, y]
				handLms = results.multi_hand_landmarks[0]
				coords = []
									
				for lm in handLms.landmark:
					coords.append([round(lm.x, 3), round(lm.y, 3)])
					
				normalized_array = normalize_landmarks(coords, label)
				
				# Enter into csv file gesture_data.csv
				# with open('/home/jb26/gesture-project-coding/gesture_data.csv', 'a', newline='') as csvfile:
					# csv_writer = csv.writer(csvfile)
					# csv_writer.writerow(normalized_array)
				
				# Change this if wanting to switch to constant feed
				r_capture = False
				
				sample = normalized_array.reshape(1,-1)
				
				# Assign to cluster and get center 
				cluster_id = kmeans.predict(sample)[0]
				center = kmeans.cluster_centers_[cluster_id]
				
				# Distance from cluster to center
				distance = np.linalg.norm(sample - center)
				
				# Check how far away current hand is from thresholds
				if thresholds[cluster_id] - distance < .25:
					current_gesture = "Unknown"
				else:
					current_gesture = gestures[cluster_id]
					
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
