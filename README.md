# Smart Online Attendance System

A Python attendance system that uses a webcam to recognize enrolled students and records their attendance in a CSV file.

## How it works
1. OpenCV reads frames from the webcam.
2. InsightFace detects each face and creates a numerical embedding for it.
3. The embedding is compared with the saved embeddings of enrolled students.
4. If the closest match is under the threshold, the student's name and time are saved to `attendance.csv`. Each student is recorded once per run.
5. Unknown faces are shown in red and are not recorded.

## Setup
You need Python 3 and a webcam.

```
pip install -r requirements.txt
```
The first run downloads the InsightFace model, so it needs an internet connection.

## Add students
Create a folder named `known_faces` next to `attendance.py`. Add one clear photo per student, named after the student, for example `Aisha.jpg`. The student photos are not included in this repository because they are private.

## Run
```
python attendance.py
```
Place your face inside the yellow box. Press **q** to quit.

## Settings
- `THRESHOLD`: maximum distance for a match. Lower values are stricter.
- `FRAME_SKIP`: process every nth frame. Higher values run faster but check less often.

## Known limitations
- Attendance is recorded once per run. Running the script again adds the students again.
- Results depend on lighting and photo quality.
- This runs on your computer with a webcam. It is not a website.
