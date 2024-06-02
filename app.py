import bcrypt
from flask import Flask, render_template, Response, request
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
 #import pygame
from flask import Flask, render_template, url_for, flash, redirect

from flask_bcrypt import Bcrypt
from flask_login import UserMixin, login_user, current_user, logout_user, login_required, LoginManager
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from mysql.connector import Error
import mysql.connector
import os
import psycopg2


app = Flask(__name__)
bcrypt = Bcrypt(app)
# Initialize LoginManager
login_manager = LoginManager(app)
login_manager.login_view = 'login'
app.config['SECRET_KEY'] = os.urandom(24)
mydb = psycopg2.connect(database="db_5ti8",
                        user="db_5ti8_user",
                        password="W6yEMZsdusTqswR358q2Td6vd9b72BnG",
                        host="dpg-cpdnv37sc6pc7394u6r0-a.oregon-postgres.render.com", port="5432")

# Create cursor object

mycursor = mydb.cursor()

class User(UserMixin):
    def __init__(self, id, username, email, password):
        self.id = id
        self.username = username
        self.email = email
        self.password = password

    @staticmethod
    def get(user_id):
        query = "SELECT * FROM users WHERE id = %s"
        mycursor.execute(query, (user_id,))
        user = mycursor.fetchone()
        if user:
            return User(user[0], user[1], user[2], user[3])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)







class RegistrationForm(FlaskForm):
    username = StringField('username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('email', validators=[DataRequired(), Email()])
    password = StringField('password', validators=[DataRequired()])
    confirm_password = StringField('confirm_password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        # Query to check if the username exists
        query = "SELECT username FROM users WHERE username = %s"
        mycursor.execute(query, (username.data,))
        user = mycursor.fetchone()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        # Query to check if the email exists
        query = "SELECT email FROM users WHERE email = %s"
        mycursor.execute(query, (email.data,))
        user = mycursor.fetchone()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
        mycursor.execute(query, (form.username.data, form.email.data, hashed_password))
        mydb.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        query = "SELECT * FROM users WHERE email = %s"
        mycursor.execute(query, (form.email.data,))
        user = mycursor.fetchone()
        print(user)
        print(user[0])
        if user and user[2] == form.password.data :
            print('You are logged in')
            return redirect(url_for('index'))
        else:
            print('Login Unsuccessful. Please check email and password')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))



#pygame.mixer.init()

#def play_alert_sound():
 #   pygame.mixer.music.load("alert_sound.mp3")  # Replace "alert_sound.mp3" with your sound file
  #  pygame.mixer.music.play()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
])

#def stop_alert():
 #   pygame.mixer.music.stop()  # Assuming you're using pygame.mixer.music for playing the alert sound

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)


# Load pre-trained model
model = CustomModel()
path = 'b.pt'
model.load_state_dict(torch.load(path))
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
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face
        face_img = frame[y:y + h, x:x + w]

        # Pre-process the face for model input
        face_tensor = transform(face_img).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(face_tensor)
            _, predicted_class = torch.max(output, 1)

        # Convert prediction to string
        if predicted_class.item() == 0:
            prediction_text = "Drowsy"
            consecutive_drowsy_frames += 1

            if consecutive_drowsy_frames >= 10 and not alert_triggered:
                #play_alert_sound()
                #alert_triggered = True
                consecutive_drowsy_frames = 0


        else:
            prediction_text = "Not Drowsy"
            consecutive_drowsy_frames = 0

            if alert_triggered:
                # Stop the alert if it was triggered
                #stop_alert()
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
     app.run(debug=True, host='0.0.0.0', port=5000)
