"""
assets/auto_capture.py - the rules for capturing faces automatically (no clicking).

Kept separate from Streamlit so they are easy to test and explain.
The live camera sends ~2 frames per second; these functions decide what to do
with each frame.
"""

import numpy as np

from assets.face_engine import MATCH_THRESHOLD, best_match

# -------- SETTINGS --------
FRAME_INTERVAL_MS = 400   # camera sends a frame at most every 0.4 s
CAPTURE_GAP = 1.5         # registration: seconds between two automatic photos
MIN_POSE_CHANGE = 0.95    # registration: new photo must be LESS similar than this to earlier ones
MAX_WAIT_FOR_CHANGE = 5   # registration: if the person doesn't move for 5 s, take the photo anyway
CONFIRM_FRAMES = 2        # attendance: same person in this many frames in a row before marking
ATTENDANCE_PAUSE = 4      # attendance: seconds to wait after marking, before the next person


# ======================================================================
# Registration
# ======================================================================

def check_registration_photo(emb, taken, ids, embs):
    """
    Checks used by BOTH auto and manual registration.
    emb   : embedding of the new photo
    taken : embeddings of photos already taken for this person
    ids, embs : everyone already stored
    Returns (ok, message, existing_person_id).
    existing_person_id is set when the face is already registered (duplicate).
    """

    # duplicate: is this face already registered?
    person_id, score = best_match(emb, ids, embs)

    if person_id is not None and score >= MATCH_THRESHOLD:
        return False, "This face is already registered.", person_id

    # all photos must show the same person
    if taken and max(float(t @ emb) for t in taken) < MATCH_THRESHOLD:
        return False, "This does not look like the same person as the first photo.", None

    return True, "ok", None


def auto_registration_ready(emb, taken, last_capture, now):
    """
    Extra rules only for AUTO registration.
    Returns a message when we should wait, otherwise None.
    """

    if not taken:
        return None

    waited = now - last_capture

    if waited < CAPTURE_GAP:
        return "Hold on..."

    # the photos should be a little different (head turned / moved),
    # otherwise 3 identical photos would be saved in 3 seconds.
    # After MAX_WAIT_FOR_CHANGE seconds we accept it anyway, so it never gets stuck.
    too_similar = max(float(t @ emb) for t in taken) >= MIN_POSE_CHANGE

    if too_similar and waited < CAPTURE_GAP + MAX_WAIT_FOR_CHANGE:
        return "Turn your head slightly or move a little."

    return None


# ======================================================================
# Attendance
# ======================================================================

def new_watch_state():
    """What the attendance camera remembers between frames."""
    return {"last_id": None, "count": 0, "pause_until": 0.0}


def attendance_step(state, person_id, now):
    """
    Decide what to do with one recognised (or unknown) frame.
    person_id : recognised person, or None for unknown / no face
    now       : time.time()
    Returns one of:
      "paused"     - we just marked someone, ignore frames for a moment
      "unknown"    - nobody recognised
      "confirming" - recognised, but wait for the next frame to be sure
      "mark"       - recognised CONFIRM_FRAMES times in a row: mark attendance now
    """

    if now < state["pause_until"]:
        return "paused"

    if person_id is None:
        state["last_id"], state["count"] = None, 0
        return "unknown"

    if person_id == state["last_id"]:
        state["count"] += 1
    else:
        state["last_id"], state["count"] = person_id, 1

    if state["count"] < CONFIRM_FRAMES:
        return "confirming"

    # marked: start the pause and forget this person
    state["last_id"], state["count"] = None, 0
    state["pause_until"] = now + ATTENDANCE_PAUSE

    return "mark"


def seconds_paused(state, now):
    return max(0, int(np.ceil(state["pause_until"] - now)))
