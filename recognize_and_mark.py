import cv2
import pickle
import pandas as pd
from datetime import datetime, timedelta
from utils import extract_face, get_embedding
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

# Load embeddings
with open("embeddings/embeddings.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

# Attendance file
ATTENDANCE_FILE = "attendance/attendance.csv"

# Create CSV if missing or empty
if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# Open webcam
cap = cv2.VideoCapture(0)
start_time = time.time()
MAX_RUNTIME = 15  # seconds

print("Camera started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    elapsed = time.time() - start_time
    if elapsed > MAX_RUNTIME:
        cv2.putText(frame, "No face detected. Try again later.",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        cv2.imshow("Face Attendance System", frame)
        cv2.waitKey(2000)
        break

    face = extract_face(frame)

    if face is None:
        cv2.putText(frame, "No face detected. Please look at camera.",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        cv2.imshow("Face Attendance System", frame)
        cv2.waitKey(1)
        continue

    embedding = get_embedding(face)
    identified_name = None
    max_similarity = 0.0

    for person, embeds in known_embeddings.items():
        for known_embed in embeds:
            similarity = cosine_similarity(
                [embedding], [known_embed]
            )[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                identified_name = person

    # ---------- REGISTERED FACE ----------
    if max_similarity >= 0.6:
        now = datetime.now()
        df = pd.read_csv(ATTENDANCE_FILE)

        # Check last attendance time
        user_records = df[df["Name"] == identified_name]

        if not user_records.empty:
            last_row = user_records.iloc[-1]
            last_time = datetime.strptime(
                f"{last_row['Date']} {last_row['Time']}",
                "%Y-%m-%d %H:%M:%S"
            )

            if now - last_time < timedelta(hours=1):
                cv2.putText(frame,
                            "Attendance already marked within last hour",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.imshow("Face Attendance System", frame)
                cv2.waitKey(2000)
                print("Attendance denied (1-hour rule)")
                break

        # Mark attendance
        df.loc[len(df)] = [
            identified_name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ]
        df.to_csv(ATTENDANCE_FILE, index=False)

        cv2.putText(frame, f"Attendance marked: {identified_name}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        cv2.imshow("Face Attendance System", frame)
        cv2.waitKey(1500)
        print(f"Attendance marked for {identified_name}")
        break

    # ---------- UNREGISTERED FACE ----------
    else:
        cv2.putText(frame,
                    "Face not registered. Attendance cannot be marked.",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.imshow("Face Attendance System", frame)
        cv2.waitKey(2000)
        break

cap.release()
cv2.destroyAllWindows()
