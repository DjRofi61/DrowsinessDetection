import bcrypt
from flask import Flask, render_template, Response, request
from PIL import Image
import cv2
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
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for RGB images
])

def stop_alert():
   pygame.mixer.music.stop()  # Assuming you're using pygame.mixer.music for playing the alert sound


class CustomMobileNetv2(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.mnet = mobilenet_v2(pretrained=True)
        self.mnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.freeze()

        self.mnet.classifier = nn.Sequential(
            nn.Linear(1280, output_size),
            nn.Dropout(0.1),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        return self.mnet(x)

    def freeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = True


# Load pre-trained model
model = CustomMobileNetv2(2)
path = 'modelmobilenet (2).pt'
checkpoint = torch.load(path, map_location=torch.device('cpu'))

# Load the state dictionary
model_state_dict = checkpoint['model_state_dict']
# Load state dictionary into the model
model.load_state_dict(model_state_dict)
model.eval()

# Initialize webcam
cap = None

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

is_detection_started = False

consecutive_drowsy_frames = 0
alert_triggered = False
def predict_drowsiness(frame):
    global consecutive_drowsy_frames, alert_triggered
    delay_frames = 15  # Set a delay of 30 frames (adjust as needed)
    delay_counter = 0



    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
        if predicted_class.item() == 0:
            prediction_text = "Drowsy"
            consecutive_drowsy_frames += 1

            if consecutive_drowsy_frames >= 10 and not alert_triggered:
                # play_alert_sound()
                # alert_triggered = True
                consecutive_drowsy_frames = 0

        else:
            prediction_text = "Not Drowsy"
            consecutive_drowsy_frames = 0

            if alert_triggered:
                # Stop the alert if it was triggered
                # stop_alert()
                alert_triggered = False  # Reset alert_triggered flag

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Write prediction result on the frame
        cv2.putText(frame, prediction_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

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


@app.route('/index')
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



    app.run(debug=True)



