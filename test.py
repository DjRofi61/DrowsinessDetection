import bcrypt
from flask import Flask, render_template, Response, request
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, url_for, flash, redirect
from flask_bcrypt import Bcrypt
from flask_login import UserMixin, login_user, current_user, logout_user, login_required
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
import os
import psycopg2

app = Flask(__name__)
bcrypt = Bcrypt(app)
app.config['SECRET_KEY'] = os.urandom(24)

# PostgreSQL database configuration
mydb = psycopg2.connect(
    database="db_5ti8",
    user="db_5ti8_user",
    password="W6yEMZsdusTqswR358q2Td6vd9b72BnG",
    host="dpg-cpdnv37sc6pc7394u6r0-a.oregon-postgres.render.com",
    port="5432"
)

# Create cursor object
mycursor = mydb.cursor()

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
        if user and bcrypt.check_password_hash(user[3], form.password.data):  # Assuming password is the 4th column in the users table
            print('You are logged in')
            return redirect(url_for('index'))
        else:
            print('Login Unsuccessful. Please check email and password')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global is_detection_started, cap
    data = request.json
    if data.get('start') == 'true':
        is_detection_started = True
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