"""
Smart Online Attendance System - web version (Streamlit).

Same recognition logic as attendance.py, but the photo comes from the
browser camera instead of cv2.VideoCapture, so it can run on Streamlit Cloud.

Put one photo per student in the known_faces folder, named by the student
(e.g. Aisha.jpg). Students can also be enrolled from the Enroll tab, but
those only last for the current browser session.
"""
import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from insightface.app import FaceAnalysis

import db

KNOWN_FACES_DIR = "known_faces"
MODEL_NAME = "buffalo_l"   # if the app runs out of memory online, try "buffalo_s"
DEFAULT_THRESHOLD = 1.25   # same value as attendance.py

st.set_page_config(page_title="Smart Online Attendance", layout="centered")


@st.cache_resource(show_spinner="Loading face model (the first run takes a minute)...")
def load_model():
    model = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=-1, det_size=(640, 640))
    return model


def decode(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@st.cache_resource(show_spinner="Loading enrolled students...")
def load_known(_model):
    embeddings, names = [], []
    if os.path.isdir(KNOWN_FACES_DIR):
        for filename in sorted(os.listdir(KNOWN_FACES_DIR)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(KNOWN_FACES_DIR, filename))
            if img is None:
                continue
            faces = _model.get(img)
            if not faces:
                continue
            embeddings.append(faces[0].embedding)
            names.append(os.path.splitext(filename)[0])
    return embeddings, names


@st.cache_resource(show_spinner=False)
def init_database():
    try:
        db.init_db()
        return True
    except Exception as err:
        print(f"[WARNING] MySQL not available: {err}")
        return False


model = load_model()
DB_OK = init_database()
folder_embeddings, folder_names = load_known(model)

if "extra_embeddings" not in st.session_state:
    st.session_state.extra_embeddings = []
    st.session_state.extra_names = []
if "records" not in st.session_state:
    st.session_state.records = []

known_embeddings = folder_embeddings + st.session_state.extra_embeddings
known_names = folder_names + st.session_state.extra_names


def mark(name):
    """Save attendance. Returns True if newly marked, False if already marked."""
    if DB_OK:
        return db.mark_attendance(name)
    if name in {r["Name"] for r in st.session_state.records}:
        return False
    st.session_state.records.append({"Name": name, "Time": time.strftime("%H:%M:%S")})
    return True


def current_records():
    if DB_OK:
        return db.fetch_today()
    return st.session_state.records


st.title("Smart Online Attendance")
st.caption("Take a photo. Enrolled students in the frame are marked present once.")

threshold = st.sidebar.slider(
    "Match threshold", 0.5, 30.0, DEFAULT_THRESHOLD, 0.05,
    help="Maximum distance to accept a match. Lower is stricter.",
)
st.sidebar.write(f"Enrolled students: {len(known_names)}")
if known_names:
    st.sidebar.write(", ".join(known_names))
if DB_OK:
    st.sidebar.success("Saving attendance to MySQL")
else:
    st.sidebar.warning("MySQL is not connected. Attendance is kept for this session only.")
    st.sidebar.caption("Add a [mysql] section to the app's secrets to connect.")

tab_attendance, tab_enroll = st.tabs(["Take attendance", "Enroll a student"])

with tab_attendance:
    if not known_names:
        st.info("No students are enrolled yet. Add one in the Enroll tab first.")
    photo = st.camera_input("Look at the camera", key="attendance_photo")
    if photo is not None and known_names:
        frame = decode(photo)
        faces = model.get(frame)
        if not faces:
            st.warning("No face found. Move closer and try again.")
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            distances = [np.linalg.norm(face.embedding - k) for k in known_embeddings]
            best = int(np.argmin(distances))
            if distances[best] < threshold:
                name = known_names[best]
                color = (0, 200, 0)
                label = name
                try:
                    newly_marked = mark(name)
                except Exception as err:
                    newly_marked = None
                    st.error(f"Could not save attendance: {err}")
                if newly_marked:
                    st.success(f"Marked present: {name}")
                elif newly_marked is False:
                    st.info(f"{name} is already marked.")
            else:
                color, label = (0, 0, 220), "Unknown"
                st.error("Face not recognized.")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    st.subheader("Today's attendance")
    try:
        records = current_records()
    except Exception as err:
        records = []
        st.error(f"Could not read attendance: {err}")
    if records:
        table = pd.DataFrame(records)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download attendance.csv",
            table.to_csv(index=False).encode("utf-8"),
            file_name="attendance.csv",
            mime="text/csv",
        )
        if not DB_OK and st.button("Clear attendance"):
            st.session_state.records = []
            st.rerun()
    else:
        st.write("Nobody has been marked yet.")

with tab_enroll:
    st.write("Enrolled students last for this browser session only. "
             "For permanent students, add photos to the known_faces folder.")
    new_name = st.text_input("Student name", key="enroll_name")
    new_photo = st.camera_input("Photo of the student", key="enroll_photo")
    if st.button("Enroll student"):
        if not new_name.strip():
            st.warning("Enter a name first.")
        elif new_photo is None:
            st.warning("Take a photo first.")
        else:
            faces = model.get(decode(new_photo))
            if not faces:
                st.error("No face found in that photo. Try again.")
            else:
                st.session_state.extra_embeddings.append(faces[0].embedding)
                st.session_state.extra_names.append(new_name.strip())
                st.success(f"Enrolled {new_name.strip()}.")
                st.rerun()
