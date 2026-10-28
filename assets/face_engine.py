"""
assets/face_engine.py - everything about faces.

Uses InsightFace "buffalo_s" on CPU:
  * a detector finds faces in the photo
  * a recognizer turns a face into a 512-number "embedding"
Two photos of the same person give embeddings that point in almost the same
direction, so we compare them with cosine similarity (dot product of normed vectors).
"""

import os
import shutil
import urllib.request
import zipfile

import cv2
import numpy as np

# -------- SETTINGS --------
MATCH_THRESHOLD = 0.5    # cosine similarity: >= this means "same person"
DARK_THRESHOLD = 40      # average brightness (0-255) below this = too dark
MIN_FACE_SIZE = 80       # face box smaller than this (pixels) = too far away

# -------- MODEL FILES --------
# buffalo_s has 5 model files, but we only need 2 of them:
#   det_500m.onnx  -> face detection
#   w600k_mbf.onnx -> face embeddings (recognition)
# Downloading only these keeps memory low on Streamlit Cloud.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder (one above assets/)
MODEL_DIR = os.path.join(BASE_DIR, "models", "buffalo_s")
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_s.zip"
NEEDED_FILES = ["det_500m.onnx", "w600k_mbf.onnx"]
HOME_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_s")


def model_files_present(folder):
    return all(os.path.exists(os.path.join(folder, f)) for f in NEEDED_FILES)


def download_model():
    """Put the 2 model files into models/buffalo_s (only the first time)."""

    if model_files_present(MODEL_DIR):
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    # already downloaded by another InsightFace project on this computer? copy it
    if model_files_present(HOME_MODEL_DIR):
        for f in NEEDED_FILES:
            shutil.copy(os.path.join(HOME_MODEL_DIR, f), os.path.join(MODEL_DIR, f))
        return

    # otherwise download the zip (~120 MB) and extract only the 2 files we need
    zip_path = MODEL_DIR + ".zip"
    urllib.request.urlretrieve(MODEL_URL, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            name = os.path.basename(member)
            if name in NEEDED_FILES:
                # write to a temp name first so a broken download is never used
                tmp_path = os.path.join(MODEL_DIR, name + ".part")
                with z.open(member) as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(tmp_path, os.path.join(MODEL_DIR, name))

    os.remove(zip_path)


def load_model():
    """Load InsightFace (CPU only). The app wraps this in @st.cache_resource."""

    from insightface.app import FaceAnalysis

    download_model()

    model = FaceAnalysis(
        name=MODEL_DIR,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]
    )
    model.prepare(ctx_id=-1, det_size=(640, 640))

    return model


# -------- IMAGE HELPERS --------

def decode_image(image_bytes):
    """Bytes from st.camera_input -> OpenCV BGR image."""

    data = np.frombuffer(image_bytes, dtype=np.uint8)

    return cv2.imdecode(data, cv2.IMREAD_COLOR)


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def normalize_lighting(img):
    """CLAHE on the L (lightness) channel of LAB, so colours are not changed."""

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def is_too_dark(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray.mean() < DARK_THRESHOLD


# -------- FACE -> EMBEDDING --------

def get_face(model, img):
    """
    Find exactly one usable face in the photo.
    Returns (face, message). face is None when the photo can't be used; message says why.
    face.normed_embedding is the 512-d embedding, face.bbox the box, face.kps the 5 landmarks.
    """

    if img is None:
        return None, "Could not read the photo, please try again."

    if is_too_dark(img):
        return None, "Too dark, please improve lighting."

    faces = model.get(normalize_lighting(img))

    if len(faces) == 0:
        return None, "No face found. Look straight at the camera."

    if len(faces) > 1:
        return None, "More than one face found. Only one person at a time."

    face = faces[0]
    x1, y1, x2, y2 = face.bbox

    if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
        return None, "Face is too small. Move closer to the camera."

    return face, "ok"


def get_face_embedding(model, img):
    """Same as get_face, but returns (embedding, message)."""

    face, message = get_face(model, img)

    if face is None:
        return None, message

    return face.normed_embedding, message


def registration_checks(face, img):
    """
    Extra checks for registration photos (same rules as the local app).
    Returns a message when the photo should not be used, otherwise None.
    """

    frame_w = img.shape[1]
    x1, y1, x2, y2 = face.bbox

    # ---- face centred ----
    face_center_x = (x1 + x2) / 2
    if abs(face_center_x - frame_w / 2) > frame_w * 0.25:
        return "Center your face in the camera."

    # ---- head straight (eyes level) ----
    left_eye, right_eye = face.kps[0], face.kps[1]
    eye_dist = abs(right_eye[0] - left_eye[0])

    if eye_dist == 0 or abs(right_eye[1] - left_eye[1]) > eye_dist * 0.25:
        return "Keep your head straight (don't tilt it)."

    return None


def best_match(embedding, ids, embeddings):
    """
    Compare one embedding with all stored ones.
    ids[i] is the person who owns embeddings[i] (a person has several rows).
    Returns (person_id, best_score) or (None, 0.0) when nothing is stored.
    """

    if len(embeddings) == 0:
        return None, 0.0

    scores = embeddings @ embedding      # cosine similarity for every stored photo

    best_id, best_score = None, -1.0

    # best score per person
    for person_id in np.unique(ids):

        score = float(scores[ids == person_id].max())

        if score > best_score:
            best_id, best_score = person_id, score

    return str(best_id), best_score
