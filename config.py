import os

# Path to employees images folder (each file name is used as employee name)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMPLOYEES_DIR = os.path.join(BASE_DIR, "employees")
DB_PATH = os.path.join(BASE_DIR, "database.db")

# DeepFace model selection: 'Facenet' or 'SFace' (SFace is lighter)
MODEL_NAME = "Facenet"

# Metric to use for comparing embeddings. Use 'cosine' (works well on CPU)
DISTANCE_METRIC = "cosine"

# Similarity threshold (0..1). If confidence < THRESHOLD, classify as UNKNOWN
# For cosine similarity we compute sim = 1 - cosine_distance, so higher is better.
THRESHOLD = 0.65

# Whether to play a beep on unknown detection (True/False)
SOUND_ON_UNKNOWN = True

# Frame resize width to speed up processing (set None to use original size)
FRAME_WIDTH = 800

# Whether to show the OpenCV window
SHOW_WINDOW = True

# Camera read timeout / read attempts (for RTSP stability)
CAMERA_READ_RETRIES = 3

# Presence monitoring: if a known employee is not seen for this many seconds,
# we consider them absent and stop the active work timer. Increase if the
# camera is noisy or detection flickers.
PRESENCE_TIMEOUT = 5  # seconds

# Whether to record working sessions into the database (sessions table)
RECORD_SESSIONS = True

# After this many seconds of continuous absence, raise an alert (beep + message)
# Set to 60 for one minute
ABSENCE_ALERT_SECONDS = 60
