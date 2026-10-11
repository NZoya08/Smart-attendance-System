"""
Smart Online Attendance System
Recognizes enrolled students from a webcam feed and records attendance to attendance.csv.

Usage:
    1. Put one photo per student in the known_faces folder, named by the student (e.g. Aisha.jpg).
    2. Run: python attendance.py
    3. Press q to quit.
"""
import cv2
import os
import csv
import time
import numpy as np
import insightface

# -----------------------------
# CONFIG
# -----------------------------
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
THRESHOLD = 1.25   # Maximum distance to accept a match
FRAME_SKIP = 2     # Process every 2nd frame for speed

# -----------------------------
# LOAD INSIGHTFACE MODEL
# -----------------------------
print("[INFO] Loading InsightFace model...")
app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# -----------------------------
# LOAD KNOWN FACES
# -----------------------------
known_embeddings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        name = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join(KNOWN_FACES_DIR, filename))
        if img is None:
            print(f"[WARNING] Could not read {filename}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"[WARNING] No face found in {filename}")
            continue

        known_embeddings.append(faces[0].embedding)
        known_names.append(name)
        print(f"[LOADED] {name}")

if not known_names:
    print("[ERROR] No student photos found in the known_faces folder.")
    exit()

print(f"[INFO] {len(known_names)} students loaded.")

# -----------------------------
# ATTENDANCE FILE SETUP
# -----------------------------
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Time"])

# Students already marked in this run (each person is recorded once)
marked = set()

# -----------------------------
# START WEBCAM
# -----------------------------
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam failed to open")
    exit()

frame_count = 0

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Failed to capture frame")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    h, w = frame.shape[:2]

    # Yellow guide box
    box_x1, box_y1 = int(w * 0.25), int(h * 0.25)
    box_x2, box_y2 = int(w * 0.75), int(h * 0.75)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Face must be inside the yellow box
        if not (box_x1 < x1 < box_x2 and box_y1 < y1 < box_y2):
            cv2.putText(frame, "Place face inside the box",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            continue

        # Find the closest known student
        distances = [np.linalg.norm(face.embedding - known) for known in known_embeddings]
        min_dist = min(distances)
        name = known_names[distances.index(min_dist)]

        if min_dist < THRESHOLD:
            # Green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Record attendance once per student per run
            if name not in marked:
                with open(ATTENDANCE_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([name, time.strftime("%H:%M:%S")])
                marked.add(name)
                print(f"[MARKED] {name}")
        else:
            # Red box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
