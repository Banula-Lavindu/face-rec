import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import date
import base64
from deepface import DeepFace
import os
import time
import shutil
from scipy.spatial import distance
import json
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///face_recognition.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

logging.basicConfig(level=logging.INFO)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    phone = db.Column(db.String(50), nullable=False)
    loyalty_points = db.Column(db.Integer, default=0)
    attendance_count = db.Column(db.Integer, default=0)
    encodings = db.Column(db.Text, nullable=False)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)

@app.route('/delete_client/<int:client_id>', methods=['POST'])
@login_required
def delete_client(client_id):
    client = Client.query.get(client_id)
    if client:
        db.session.delete(client)
        db.session.commit()
        flash('Client deleted successfully')
    return redirect(url_for('index'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]
    return face

@app.route('/')
@login_required
def index():
    clients = Client.query.all()
    return render_template('index.html', clients=clients)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        img_data = request.form['img_data']

        try:
            img_data = base64.b64decode(img_data.split(',')[1])
        except IndexError:
            flash('Failed to decode image data. Please try again.')
            return redirect(url_for('register'))

        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face = crop_face(img)
        if face is None:
            flash('No face detected. Please try again.')
            return redirect(url_for('register'))

        try:
            face_encoding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']
        except Exception as e:
            flash('Failed to generate face encoding. Please try again.')
            logging.error(f"DeepFace error: {e}")
            return redirect(url_for('register'))

        if Client.query.filter_by(name=name).first():
            flash('Client already exists')
            return redirect(url_for('register'))

        new_client = Client(name=name, phone=phone, encodings=json.dumps([face_encoding]))
        db.session.add(new_client)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_recognition')
@login_required
def live_recognition():
    return render_template('live_recognition.html')

@app.route('/recognize_live', methods=['POST'])
@login_required
def recognize_live():
    img_data = request.json['img_data']
    img_data = base64.b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face = crop_face(frame)
    if face is None:
        return jsonify({'recognized_name': 'No Face Detected'})

    try:
        face_encoding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']
    except Exception as e:
        logging.error(f"DeepFace error: {e}")
        return jsonify({'recognized_name': 'Error Processing Face'})

    if not np.isfinite(face_encoding).all():
        logging.error(f"Invalid face encoding: {face_encoding}")
        return jsonify({'recognized_name': 'Invalid Face Encoding'})

    recognized_name = "Unknown"
    min_distance = float('inf')

    clients = Client.query.all()
    for client in clients:
        client_encodings = json.loads(client.encodings)
        for encoding in client_encodings:
            if not np.isfinite(encoding).all():
                logging.error(f"Invalid client encoding for {client.name}: {encoding}")
                continue
            dist = distance.euclidean(face_encoding, encoding)
            if dist < min_distance:
                min_distance = dist
                recognized_name = client.name

    if recognized_name != "Unknown":
        client = Client.query.filter_by(name=recognized_name).first()
        if client:
            client_encodings = json.loads(client.encodings)
            client_encodings.append(face_encoding)
            if len(client_encodings) > 10:
                client_encodings = client_encodings[-10:]
            client.encodings = json.dumps(client_encodings)
            db.session.commit()

            today = date.today()
            attendance = Attendance.query.filter_by(client_id=client.id, date=today).first()
            if not attendance:
                new_attendance = Attendance(client_id=client.id, date=today)
                db.session.add(new_attendance)
                client.loyalty_points += 20
                client.attendance_count += 1
                db.session.commit()
            return jsonify({
                'recognized_name': recognized_name,
                'attendance_count': client.attendance_count,
                'loyalty_points': client.loyalty_points
            })

    return jsonify({'recognized_name': 'Unknown'})

@app.route('/detect_face', methods=['POST'])
def detect_face():
    img_data = request.json['img_data']
    img_data = base64.b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return jsonify({'detected': False})

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    _, buffer = cv2.imencode('.jpg', face)
    face_data = base64.b64encode(buffer).decode('utf-8')
    face_rect = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
    return jsonify({'detected': True, 'face_data': 'data:image/jpeg;base64,' + face_data, 'face_rect': face_rect})

@app.route('/detect_and_recognize_face', methods=['POST'])
def detect_and_recognize_face():
    img_data = request.json['img_data']
    img_data = base64.b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return jsonify({'recognized_name': 'No Face Detected'})

    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]
    _, buffer = cv2.imencode('.jpg', face)
    face_data = base64.b64encode(buffer).decode('utf-8')

    try:
        face_encoding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']
    except Exception as e:
        logging.error(f"DeepFace error: {e}")
        return jsonify({'recognized_name': 'Error Processing Face'})

    if not np.isfinite(face_encoding).all():
        logging.error(f"Invalid face encoding: {face_encoding}")
        return jsonify({'recognized_name': 'Invalid Face Encoding'})

    recognized_name = "Unknown"
    min_distance = float('inf')

    clients = Client.query.all()
    for client in clients:
        client_encodings = json.loads(client.encodings)
        for encoding in client_encodings:
            if not np.isfinite(encoding).all():
                logging.error(f"Invalid client encoding for {client.name}: {encoding}")
                continue
            dist = distance.euclidean(face_encoding, encoding)
            if dist < min_distance:
                min_distance = dist
                recognized_name = client.name

    if recognized_name != "Unknown":
        client = Client.query.filter_by(name=recognized_name).first()
        if client:
            client_encodings = json.loads(client.encodings)
            client_encodings.append(face_encoding)
            if len(client_encodings) > 10:
                client_encodings = client_encodings[-10:]
            client.encodings = json.dumps(client_encodings)
            db.session.commit()

            today = date.today()
            attendance = Attendance.query.filter_by(client_id=client.id, date=today).first()
            if not attendance:
                new_attendance = Attendance(client_id=client.id, date=today)
                db.session.add(new_attendance)
                client.loyalty_points += 20
                client.attendance_count += 1
                db.session.commit()
            face_rect = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            return jsonify({
                'recognized_name': recognized_name,
                'attendance_count': client.attendance_count,
                'loyalty_points': client.loyalty_points,
                'face_rect': face_rect
            })

    return jsonify({'recognized_name': 'Unknown'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
