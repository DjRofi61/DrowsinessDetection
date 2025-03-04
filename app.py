import numpy as np  # Add this import
from flask import Flask, render_template, Response, request
from PIL import Image
import cv2
import torch.nn.functional as F  # Add this import
import torch
import torch.nn as nn
from torchvision import models, transforms
#import pygame
from flask import Flask, render_template, url_for, flash, redirect
from torchvision.models import mobilenet_v2
from flask_bcrypt import Bcrypt
from PIL import Image
import os
app = Flask(__name__)
bcrypt = Bcrypt(app)
# Initialize LoginManager

# Configuration for file uploads and results
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'



#pygame.mixer.init()
"""
def play_alert_sound():
    pygame.mixer.music.load("alert_sound.mp3")  # Replace "alert_sound.mp3" with your sound file
    pygame.mixer.music.play()

def stop_alert():
    pygame.mixer.music.stop()  # Assuming you're using pygame.mixer.music for playing the alert sound"""

# Define the model
class CustomModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomModel, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        self.mobilenet_v2.classifier = nn.Identity()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1280, 8)
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.mobilenet_v2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# Instantiate the model
model = CustomModel(num_classes=3)
model.load_state_dict(torch.load('b.pth'), strict=False)
model.eval()

# Load YOLO model for face detection
#yolo_model = YOLO("yolov8-face/yolov8n-face.pt")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict')
def predict():
    predefined_images = [
        'images/yawnMaleFemale420.jpg',
        'images/yawnMaleFemale600.jpg',
        'images/yawnFemale9816.jpg',
        'images/yawnMale13932.jpg',
        'images/yawnMale16758.jpg',
        'images/yawnMale22998.jpg',
        'images/yawnMaleFemale3730.jpg',
        'images/yawnMale25944.jpg',
        'images/yawnMale33732.jpg'
    ]
    return render_template('predict.html', predefined_images=predefined_images)
@app.route('/get_result_image', methods=['POST'])
def get_result_image():
    image_path = request.json.get('image_path')
    if image_path:
        result_filename = os.path.basename(image_path)
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        if os.path.exists(result_filepath):
            return {'result_filename': result_filename}
        else:
            return {'error': 'Result image not found'}, 404
    return {'error': 'Invalid request'}, 400


""""@app.route('/predict_predefined', methods=['POST'])
def predict_predefined():
    image_path = request.form.get('image_path')
    if image_path:
        # Perform prediction
        result_filepath = predict_drowsiness(image_path)

        # Extract filename from path
        filename = os.path.basename(image_path)

        return render_template('predict.html', filename=filename)
    return redirect(url_for('index'))"""

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/predict_predefined', methods=['POST'])
def predict_predefined():
    image_path = request.form.get('image_path')
    if image_path:
        # Assuming the image_path is something like 'static/images/image.jpg'
        result_filename = os.path.basename(image_path)  # Extract filename from path
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)  # Path to the result image

        # Check if result image exists
        if os.path.exists(result_filepath):
            return render_template('predict.html', filename=result_filename)
        else:
            flash('Result image not found!', 'error')
            return redirect(url_for('index'))
    return redirect(url_for('index'))


def predict_drowsiness(image_path):
    alert_triggered = False
    img = Image.open(image_path).convert('RGB')
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop the face
            face_img = frame[y:y + h, x:x + w]

            # Convert the face image from NumPy array to PIL image
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            # Pre-process the face for model input
            face_tensor = transform(face_pil).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = model(face_tensor)
                _, predicted_class = torch.max(output, 1)

            # Convert prediction to string
            class_names = ['yawning', 'closeEyes', 'awake']
            prediction_text = class_names[predicted_class.item()]

            if prediction_text in ['closeEyes', 'yawning']:
                if not alert_triggered:
                    #play_alert_sound()
                    alert_triggered = True
            else:
                if alert_triggered:
                    #stop_alert()
                    alert_triggered = False

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Write prediction result on the frame
            cv2.putText(frame, prediction_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Save the result image
    result_filename = os.path.basename(image_path)
    result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_filepath, frame)

    return result_filepath

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RESULT_FOLDER']):
        os.makedirs(app.config['RESULT_FOLDER'])

    app.run()
"""
import bcrypt
from flask import Flask, render_template, Response, request
from PIL import Image
import cv2
import torch.nn.functional as F  # Add this import
import torch
import torch.nn as nn
from torchvision import models, transforms
import pygame
from flask import Flask, render_template, url_for, flash, redirect
from torchvision.models import mobilenet_v2
from flask_bcrypt import Bcrypt
from PIL import Image
from flask_login import UserMixin, login_user, current_user, logout_user, login_required, LoginManager
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from ultralytics import YOLO
import os
import psycopg2


app = Flask(__name__)
bcrypt = Bcrypt(app)
# Initialize LoginManager



pygame.mixer.init()

def play_alert_sound():
    pygame.mixer.music.load("alert_sound.mp3")  # Replace "alert_sound.mp3" with your sound file
    pygame.mixer.music.play()
bs = 128
crop_size = 224
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def stop_alert():
   pygame.mixer.music.stop()  # Assuming you're using pygame.mixer.music for playing the alert sound


# Define the model
class CustomModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomModel, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        self.mobilenet_v2.classifier = nn.Identity()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1280, 8)
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.mobilenet_v2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# Instantiate the model
model = CustomModel(num_classes=3)

path = 'b.pth'

model.load_state_dict(torch.load('b.pth'), strict=False)

model.eval()
# Set the model to evaluation mode



# Initialize webcam
cap = None

# Load Haar Cascade classifier for face detection
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load YOLO model for face detection
yolo_model = YOLO("yolov8n-face.pt")

is_detection_started = False

consecutive_drowsy_frames = 0
alert_triggered = False
def predict_drowsiness(frame):
    global alert_triggered
    delay_frames = 15  # Set a delay of 30 frames (adjust as needed)
    delay_counter = 0

    # Detect faces in the frame using YOLO
    results = yolo_model(frame)

    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])

            # Crop the face
            face_img = frame[y1:y2, x1:x2]

            # Convert the face image from NumPy array to PIL image
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            # Pre-process the face for model input
            face_tensor = transform(face_pil).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = model(face_tensor)
                _, predicted_class = torch.max(output, 1)

            # Convert prediction to string
            class_names = ['yawning', 'closeEyes', 'awake']
            prediction_text = class_names[predicted_class.item()]

            if prediction_text in ['closeEyes', 'yawning']:
                if not alert_triggered:
                    play_alert_sound()
                    alert_triggered = True
            else:
                if alert_triggered:
                    stop_alert()
                    alert_triggered = False

            # Draw rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Write prediction result on the frame
            cv2.putText(frame, prediction_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame


def generate_frames():
    global cap
    while cap.isOpened():
        # Capture frame from webcam
        ret, frame = cap.read()

        if not ret:
            break

        if is_detection_started:
            # Perform prediction on the frame
            frame_with_prediction = predict_drowsiness(frame)
        else:
            frame_with_prediction = frame

        # Encode frame as JPEG image
        ret, jpeg = cv2.imencode('.jpg', frame_with_prediction)
        frame_bytes = jpeg.tobytes()

        # Yield frame bytes to be streamed to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_detection', methods=['POST'])
def start_detection():
    global is_detection_started, cap
    # Retrieve the data sent by the client
    data = request.json

    # Check if the data contains a "start" key with value "true"
    if data.get('start') == 'true':
        # Set the detection flag to True and start the video capture
        is_detection_started = True
        cap = None
        cap = cv2.VideoCapture(0)
        print("Drowsiness detection started.")


    else:
        if data.get('start') == 'false':
         is_detection_started = False
         if cap:
            cap.release()
            cap = None

    return 'OK'


if __name__ == "__main__":



    app.run(debug=True)"""
