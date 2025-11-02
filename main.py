"""
main.py

Real-time CCTV face recognition (CPU-only) using OpenCV + DeepFace.

Features:
- Loads employee images from ./employees and computes embeddings at startup
- Captures video from webcam (index) or RTSP stream
- Detects faces with OpenCV Haar cascade
- Computes embedding for each detected face and compares with known embeddings
- Logs each detection into SQLite (auto-creates DB/table)
- Prints and optionally beeps on unknown detection
- Provides a register function to add employee images and embeddings at runtime

Functions:
- load_employee_embeddings(): reads images from employees/ and returns dict{name:embedding}
- create_db_if_needed(): creates SQLite DB and detections table if missing
- log_detection(name, camera_id, confidence): inserts a detection record
- register_employee(name, image_path): copies image into employees/ and updates embeddings
- compute_embedding(img): returns DeepFace embedding for a face image array
- recognize_face(face_img, known_embeddings, threshold): returns (name, confidence)
- run_camera(camera_source, known_embeddings): main loop, displays frames

CLI:
- --camera : 0 (webcam) or rtsp://... (RTSP)
- --threshold : similarity threshold (default from config)
- --sound : True/False to play beep on unknown
- --register : register a new employee: provide name and image path

"""

import os
import argparse
import shutil
import sqlite3
import time
from datetime import datetime

import cv2
import numpy as np
try:
    from deepface import DeepFace
except Exception as e:
    # DeepFace may import optional backends (e.g., retinaface) which require
    # the `tf-keras` package when using newer TensorFlow versions (>=2.20).
    # Provide an actionable error message instead of a long traceback.
    print("[ERROR] Failed to import DeepFace. If you see an error mentioning 'tf_keras' or 'retinaface',"
          " install the compatibility package:")
    print("    pip install tf-keras")
    print("or pin TensorFlow to an earlier version (e.g. tensorflow-cpu==2.11).")
    # Re-raise so the stack trace is visible after our explanatory message.
    raise
import imutils

import config

# Local constants
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Ensure employees dir exists
os.makedirs(config.EMPLOYEES_DIR, exist_ok=True)

# DB helpers
def create_db_if_needed(db_path: str = config.DB_PATH):
    """Create SQLite database and detections table if not exists."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_id TEXT,
            confidence REAL
        )
        """
    )
    # Optional sessions table to record working sessions per employee
    if config.RECORD_SESSIONS:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                start_time DATETIME,
                end_time DATETIME,
                duration_seconds REAL,
                camera_id TEXT
            )
            """
        )
    conn.commit()
    conn.close()


def log_detection(name: str, camera_id: str, confidence: float, db_path: str = config.DB_PATH):
    """Insert a detection record into database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT INTO detections (name, camera_id, confidence) VALUES (?, ?, ?)",
        (name, camera_id, float(confidence)),
    )
    conn.commit()
    conn.close()


def log_session(name: str, start_time: datetime, end_time: datetime, duration_seconds: float, camera_id: str, db_path: str = config.DB_PATH):
    """Insert a working session record into the sessions table."""
    if not config.RECORD_SESSIONS:
        return
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT INTO sessions (name, start_time, end_time, duration_seconds, camera_id) VALUES (?, ?, ?, ?, ?)",
        (name, start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S'), float(duration_seconds), camera_id),
    )
    conn.commit()
    conn.close()


# Embedding helpers
def compute_embedding(img_rgb: np.ndarray, model_name: str = config.MODEL_NAME):
    """Compute embedding for an image (face crop in RGB) using DeepFace.

    img_rgb: numpy array in RGB color order.
    Returns a 1D numpy array embedding.
    """
    # DeepFace expects BGR or path; when passing an array, it's treated as BGR by some versions.
    # To be safe, pass the RGB array but set enforce_detection=False for robustness.
    try:
        # Use OpenCV backend for detection inside DeepFace (lighter, CPU-friendly)
        # and set enforce_detection=False because we already pass face crops.
        rep = DeepFace.represent(img_rgb, model_name=model_name, detector_backend='opencv', enforce_detection=False)
        # DeepFace.represent returns list of dicts or a single embedding depending on version.
        # Normalize handling below.
        if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and "embedding" in rep[0]:
            emb = np.array(rep[0]["embedding"])  # older DeepFace form
        elif isinstance(rep, dict) and "embedding" in rep:
            emb = np.array(rep["embedding"])
        elif isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], (list, np.ndarray)):
            emb = np.array(rep[0])
        elif isinstance(rep, (list, np.ndarray)):
            emb = np.array(rep)
        else:
            raise ValueError("Unexpected representation return format")
        return emb
    except Exception as e:
        print(f"[WARN] compute_embedding: DeepFace.represent failed: {e}")
        return None


def load_employee_embeddings():
    """Load employee images from the employees folder and compute embeddings.

    Returns dict: {name: embedding}
    Filename without extension is used as name.
    """
    embeddings = {}
    display_names = {}
    files = [f for f in os.listdir(config.EMPLOYEES_DIR) if os.path.isfile(os.path.join(config.EMPLOYEES_DIR, f))]
    if not files:
        print("[INFO] No employee images found in employees/ folder.")
        return embeddings

    print(f"[INFO] Loading {len(files)} employee images and computing embeddings (this may take a while)...")
    for fn in files:
        path = os.path.join(config.EMPLOYEES_DIR, fn)
        name = os.path.splitext(fn)[0]
        try:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print(f"[WARN] Could not read image {path}")
                continue
            # Convert to RGB for DeepFace
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            emb = compute_embedding(img_rgb)
            if emb is not None:
                embeddings[name] = emb
                # create a nicer display name for UI
                display = sanitize_display_name(name)
                display_names[name] = display
                print(f"[INFO] Loaded embedding for: {name} (display: {display})")
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
    print(f"[INFO] Completed loading embeddings. Known employees: {len(embeddings)}")
    return embeddings, display_names


def sanitize_display_name(raw_name: str) -> str:
    """Create a shorter, human-friendly display name from a filename key.

    - replace underscores with spaces
    - remove common camera app prefixes like 'WhatsApp Image'
    - collapse multiple spaces
    - trim to a reasonable length
    """
    s = raw_name.replace('_', ' ')
    # remove common noisy prefixes
    s = s.replace('WhatsApp Image', '')
    s = s.replace('IMG', '')
    s = s.replace('Image', '')
    # remove extra spaces
    s = ' '.join(s.split())
    s = s.strip()
    # if still looks like a hash or too long, shorten
    if len(s) == 0:
        s = raw_name
    if len(s) > 20:
        s = s[:17].rstrip() + '...'
    return s


# Simple similarity using cosine distance
def cosine_similarity(a: np.ndarray, b: np.ndarray):
    if a is None or b is None:
        return 0.0
    a = a.flatten()
    b = b.flatten()
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # cosine similarity in [-1,1], normalize to [0,1]
    sim_norm = (sim + 1.0) / 2.0
    return float(sim_norm)


def recognize_face(face_rgb: np.ndarray, known_embeddings: dict, threshold: float):
    """Compare face_rgb embedding to known embeddings.

    Returns (best_name, best_confidence) where confidence is in [0,1]
    If no known embeddings, returns ("Unknown", 0.0)
    """
    if not known_embeddings:
        return "Unknown", 0.0

    emb = compute_embedding(face_rgb)
    if emb is None:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_score = 0.0
    for name, kem in known_embeddings.items():
        score = cosine_similarity(emb, kem)  # higher is better
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    else:
        return "Unknown", best_score


def register_employee(name: str, image_path: str, known_embeddings: dict, display_names: dict = None):
    """Register a new employee by copying the image into employees/ and computing embedding.

    name: display name (will be sanitized into file name)
    image_path: path to source image
    known_embeddings: dict to update in-place
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")
    # Sanitize filename
    safe_name = "_".join(name.strip().split())
    _, ext = os.path.splitext(image_path)
    dest = os.path.join(config.EMPLOYEES_DIR, safe_name + ext)
    shutil.copyfile(image_path, dest)
    # compute embedding and update
    img_bgr = cv2.imread(dest)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    emb = compute_embedding(img_rgb)
    if emb is not None:
        known_embeddings[safe_name] = emb
        # update display map if provided
        if display_names is not None:
            display_names[safe_name] = name
        print(f"[INFO] Registered employee: {safe_name} (display: {display_names.get(safe_name) if display_names else name})")
    else:
        print(f"[WARN] Registered file saved but embedding could not be computed for: {safe_name}")


def beep_alert():
    """Cross-platform beep on unknown detection (best-effort)."""
    try:
        # Windows
        import winsound

        winsound.Beep(1000, 300)
    except Exception:
        # Fallback: bell char
        print('\a', end='')


def init_presence(known_embeddings: dict):
    """Initialize presence tracking structure for known employees.

    Presence dict per employee:
      start: datetime when current session started or None
      accumulated: float seconds accumulated so far (since program start)
      last_seen: datetime of last detection
      is_present: bool
    """
    presence = {}
    for name in known_embeddings.keys():
        presence[name] = {
            'start': None,
            'accumulated': 0.0,
            'last_seen': None,
            'is_present': False,
            'absent_since': None,
            'alerted_absent': False,
        }
    return presence


def finalize_presence_for_exit(presence: dict, camera_id: str):
    """When exiting, finalize any active sessions and write to DB."""
    now = datetime.now()
    for name, info in presence.items():
        if info['is_present'] and info['start'] is not None:
            end_time = info['last_seen'] or now
            duration = (end_time - info['start']).total_seconds()
            info['accumulated'] += duration
            # log session
            try:
                log_session(name, info['start'], end_time, duration, camera_id)
            except Exception as e:
                print(f"[WARN] Failed to log session for {name}: {e}")
            info['start'] = None
            info['is_present'] = False


def seconds_to_hms(secs: float):
    secs = int(secs)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_camera(camera_source, known_embeddings, threshold: float, sound_on_unknown: bool, display_names: dict = None):
    """Main loop: capture frames, detect faces, recognize, log, and display."""
    create_db_if_needed()

    # Prepare video capture
    # camera_source can be int index or RTSP string
    cam = None
    if isinstance(camera_source, int):
        cam = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(camera_source)

    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera source: {camera_source}")

    camera_id = str(camera_source)
    print(f"[INFO] Starting camera: {camera_id}")
    # initialize presence tracking for known employees
    presence = init_presence(known_embeddings)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[WARN] Frame read failed, retrying...")
            # try a few times then break
            retries = 0
            good = False
            while retries < config.CAMERA_READ_RETRIES:
                time.sleep(0.5)
                ret, frame = cam.read()
                if ret:
                    good = True
                    break
                retries += 1
            if not good:
                print("[ERROR] Failed to read from camera. Exiting loop.")
                break

        # optional resize for speed
        orig_h, orig_w = frame.shape[:2]
        if config.FRAME_WIDTH is not None and orig_w > config.FRAME_WIDTH:
            frame = imutils.resize(frame, width=config.FRAME_WIDTH)

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        # track which known employees were recognized this frame
        recognized_this_frame = set()

        for (x, y, w, h) in faces:
            # expand box slightly
            pad = int(0.1 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face_bgr = frame[y1:y2, x1:x2]
            if face_bgr.size == 0:
                continue
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

            name, confidence = recognize_face(face_rgb, known_embeddings, threshold)
            label = f"{name} ({confidence:.2f})"

            # draw box + label (use display name if available)
            display_label = label
            if name != "Unknown" and display_names is not None:
                dname = display_names.get(name, None)
                if dname:
                    display_label = f"{dname} ({confidence:.2f})"
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ts_dt = datetime.now()
            ts = ts_dt.strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(display, ts, (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # log detection
            log_detection(name if name != "Unknown" else "Unknown", camera_id, confidence)

            # update presence if it's a known person
            if name != "Unknown":
                recognized_this_frame.add(name)
                now = ts_dt
                info = presence.get(name)
                if info is not None:
                    info['last_seen'] = now
                    # reset absent flags when seen again
                    info['absent_since'] = None
                    info['alerted_absent'] = False
                    if not info['is_present']:
                        # mark session start
                        info['start'] = now
                        info['is_present'] = True
            else:
                # alert unknown
                print(f"[ALERT] Unknown person detected at {ts} (score={confidence:.2f}) on camera {camera_id}")
                if sound_on_unknown:
                    beep_alert()

        # After processing faces, check for absences (timeout)
        now_check = datetime.now()
        for name, info in presence.items():
            if info['is_present']:
                last = info['last_seen'] or now_check
                if (now_check - last).total_seconds() > config.PRESENCE_TIMEOUT:
                    # finalize session
                    end_time = info['last_seen'] or now_check
                    duration = (end_time - info['start']).total_seconds() if info['start'] else 0.0
                    info['accumulated'] += duration
                    # log session to DB
                    try:
                        log_session(name, info['start'], end_time, duration, camera_id)
                    except Exception as e:
                        print(f"[WARN] Failed to log session for {name}: {e}")
                    print(f"[INFO] Session ended for {name}: duration {seconds_to_hms(duration)}")
                    info['start'] = None
                    info['is_present'] = False
                    # mark when the person became absent for future alerts
                    info['absent_since'] = now_check
                    info['alerted_absent'] = False

        # collect missing alerts to display prominently
        missing_alerts = []

        # check for long absences and trigger one-time alerts
        for name, info in presence.items():
            if not info['is_present'] and info.get('absent_since') is not None and not info.get('alerted_absent'):
                if (now_check - info['absent_since']).total_seconds() >= config.ABSENCE_ALERT_SECONDS:
                    print(f"[ALERT] {name} not present for {config.ABSENCE_ALERT_SECONDS} seconds on camera {camera_id}")
                    try:
                        beep_alert()
                    except Exception:
                        pass
                    info['alerted_absent'] = True
                    missing_alerts.append(f"{name} is NOT PRESENT")

        # Overlay presence summary on the display (top-left)
        y0 = 20
        for name, info in presence.items():
            # compute total including active session
            total = info['accumulated']
            if info['is_present'] and info['start'] is not None:
                total += (now_check - info['start']).total_seconds()
            status = 'Working' if info['is_present'] else 'Away'
            # compute last-seen and timeout remaining (for present users)
            last_seen = info.get('last_seen')
            last_seen_secs = None
            if last_seen is not None:
                last_seen_secs = int((now_check - last_seen).total_seconds())

            if info['is_present']:
                # seconds remaining until considered absent
                timeout_remain = int(max(0, config.PRESENCE_TIMEOUT - (last_seen_secs if last_seen_secs is not None else 0)))
                last_part = f"last:{last_seen_secs}s" if last_seen_secs is not None else ""
                # use display name if available
                disp = display_names.get(name, name) if display_names is not None else name
                text = f"{disp}: {status} {seconds_to_hms(total)} | {last_part} | timeout in {timeout_remain}s"
            else:
                disp = display_names.get(name, name) if display_names is not None else name
                text = f"{disp}: {status} {seconds_to_hms(total)}"
            cv2.putText(display, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            y0 += 22

        # Overlay missing alerts (prominent, in red)
        if missing_alerts:
            y_alert = y0 + 10
            for msg in missing_alerts:
                cv2.putText(display, msg, (10, y_alert), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                y_alert += 30

        if config.SHOW_WINDOW:
            cv2.imshow('Face Recognition', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quitting by user request (q)")
                break

    # finalize any active presence sessions on exit
    try:
        finalize_presence_for_exit(presence, camera_id)
    except Exception as e:
        print(f"[WARN] finalize_presence_for_exit failed: {e}")

    cam.release()
    cv2.destroyAllWindows()


def parse_camera_arg(cam_arg: str):
    # return int if numeric, otherwise string
    try:
        return int(cam_arg)
    except Exception:
        return cam_arg


def main():
    parser = argparse.ArgumentParser(description="Real-time CCTV Face Recognition (CPU-only)")
    parser.add_argument('--camera', required=False, default='0', help='Camera source: 0 (webcam) or rtsp://...')
    parser.add_argument('--threshold', required=False, type=float, default=config.THRESHOLD, help='Similarity threshold (0..1)')
    parser.add_argument('--sound', required=False, type=lambda x: x.lower() in ['true', '1', 'yes'], default=config.SOUND_ON_UNKNOWN, help='Play beep on unknown detections')
    parser.add_argument('--register', nargs=2, metavar=('NAME', 'IMAGE_PATH'), help='Register a new employee: NAME IMAGE_PATH')

    args = parser.parse_args()

    # Create DB if not exists
    create_db_if_needed()

    # Load known embeddings and display name map
    known_embeddings, display_names = load_employee_embeddings()

    # If register requested, do it and exit
    if args.register:
        name, image_path = args.register
        try:
            # update both known embeddings and display names map
            register_employee(name, image_path, known_embeddings, display_names)
            print("[INFO] Registration complete. You can re-run the service to include new employee embeddings.")
        except Exception as e:
            print(f"[ERROR] Registration failed: {e}")
        return

    cam = parse_camera_arg(args.camera)
    try:
        run_camera(cam, known_embeddings, args.threshold, args.sound, display_names=display_names)
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == '__main__':
    main()
