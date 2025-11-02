# Real-time CCTV Face Recognition (CPU-only)

This project captures live video (webcam or RTSP) and performs real-time face recognition using DeepFace (Facenet/SFace) on CPU.

Structure:

project/
├─ main.py
├─ config.py
├─ employees/ # employee image database (place employee images here)
├─ database.db # SQLite auto-created
├─ requirements.txt

Quick install (Windows / Linux):

1. Create a Python 3.9+ virtual environment and activate it.

On Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app (webcam):

```
python main.py --camera 0
```

3. Run with RTSP camera:

```
python main.py --camera rtsp://user:pass@ip:554/stream
```

Register a new employee image (copy image into `employees/` and update embeddings):

```
python main.py --register "John Doe" "path/to/john.jpg"
```

Notes:

- `sqlite3` is part of the Python standard library.
- Use `tensorflow-cpu` in `requirements.txt` to avoid GPU/CUDA installs.
- Drop employee images into `employees/` named like `John_Doe.jpg`. The base filename (without extension) is used as the employee name.

Functions documented in `main.py`.

## Presence monitoring

The app now tracks per-employee presence time. When a known employee is detected the timer starts. If the employee is not seen for a configurable timeout (default 5s), the session is closed and saved to the database.

Sessions are recorded in the `sessions` table in `database.db` with fields: `id`, `name`, `start_time`, `end_time`, `duration_seconds`, `camera_id`.
