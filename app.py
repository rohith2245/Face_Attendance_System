from flask import Flask, render_template, Response, redirect, url_for, request, flash, jsonify
import cv2
import pandas as pd
import os
import pickle
import base64
import numpy as np
import threading
import subprocess
from datetime import datetime, timedelta
from utils import extract_face, get_embedding
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "attendance_secret"

ATTENDANCE_FILE = "attendance/attendance.csv"

camera = None
current_frame = None
training_status = "idle"


# ---------- LOAD EMBEDDINGS ----------
def load_embeddings():
    if os.path.exists("embeddings/embeddings.pkl"):
        with open("embeddings/embeddings.pkl", "rb") as f:
            return pickle.load(f)
    return {}

known_embeddings = load_embeddings()


# ---------- CAMERA ----------
def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera


# ---------- VIDEO STREAM ----------
def generate_frames():
    global current_frame
    cam = get_camera()

    while True:
        success, frame = cam.read()
        if not success:
            break

        current_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------- HOME ----------
@app.route("/")
def home():

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        grouped = df.groupby("Date")
    else:
        grouped = []

    return render_template("index.html", grouped=grouped)


# ---------- VIDEO ----------
@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------- MARK ATTENDANCE ----------
@app.route("/mark_attendance")
def mark_attendance():

    global current_frame, known_embeddings

    if current_frame is None:
        flash("Camera not ready", "error")
        return redirect(url_for("home"))

    face = extract_face(current_frame)

    if face is None:
        flash("No face detected", "error")
        return redirect(url_for("home"))

    embedding = get_embedding(face)

    identified_name = None
    max_similarity = 0

    for person, embeds in known_embeddings.items():

        for known_embed in embeds:

            similarity = cosine_similarity(
                [embedding], [known_embed]
            )[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                identified_name = person

    if max_similarity >= 0.6:

        now = datetime.now()

        if not os.path.exists(ATTENDANCE_FILE):
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
        else:
            df = pd.read_csv(ATTENDANCE_FILE)

        user_records = df[df["Name"] == identified_name]

        if not user_records.empty:

            last_row = user_records.iloc[-1]

            last_time = datetime.strptime(
                f"{last_row['Date']} {last_row['Time']}",
                "%Y-%m-%d %H:%M:%S"
            )

            if now - last_time < timedelta(hours=1):
                flash("Attendance already marked within 1 hour", "error")
                return redirect(url_for("home"))

        df.loc[len(df)] = [
            identified_name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ]

        df.to_csv(ATTENDANCE_FILE, index=False)

        flash(f"Attendance marked for {identified_name}", "success")

    else:

        flash("Face not registered", "error")

    return redirect(url_for("home"))


# ---------- BACKGROUND TRAINING ----------
def run_training():

    global training_status, known_embeddings

    subprocess.run(["python", "train_embeddings.py"])

    known_embeddings = load_embeddings()

    training_status = "done"


# ---------- TRAIN MODEL ----------
@app.route("/train")
def train():

    global training_status

    if training_status == "running":
        flash("Training already running", "error")
        return redirect(url_for("home"))

    training_status = "running"

    thread = threading.Thread(target=run_training)
    thread.start()

    flash("Training started. Please wait about 10 seconds.", "success")

    return redirect(url_for("home"))


# ---------- REGISTER FACE ----------
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        image_data = request.form["image"]

        image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        person_folder = f"data/raw_faces/{name}"

        os.makedirs(person_folder, exist_ok=True)

        image_count = len(os.listdir(person_folder)) + 1

        save_path = f"{person_folder}/image{image_count}.jpg"

        cv2.imwrite(save_path, frame)

        flash(f"{name} registered successfully. Now click TRAIN MODEL.", "success")

        return redirect(url_for("home"))

    return render_template("register.html")


# ---------- TRAIN STATUS ----------
@app.route("/training_status")
def training_status_check():

    global training_status

    status = training_status

    # reset status so alert shows only once
    if training_status == "done":
        training_status = "idle"

    return jsonify({"status": status})


# ---------- RUN SERVER ----------
if __name__ == "__main__":
    app.run(debug=True)