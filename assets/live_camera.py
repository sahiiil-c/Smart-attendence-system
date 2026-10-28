"""
assets/live_camera.py - a live webcam that sends frames to Python by itself.

Why: st.camera_input needs a click for every photo, and cv2.VideoCapture would
open the SERVER's camera (useless on Streamlit Cloud). This small custom
component (Streamlit Components v2) opens the camera in the visitor's browser.

How it talks to Python:
  * JS -> Python: every ~0.4 s it sends one JPEG frame as a "trigger"
                  (a one-time event): {"seq": 7, "image": "data:image/jpeg;base64,..."}
  * Python -> JS: on every run Python sends `data`: the message to show, a status
                  colour, whether to keep sending, and "ack" = the last frame it handled.
    The JS waits for that ack before sending the next frame, so the server
    is never flooded with frames.

Usage inside an @st.fragment:
    frame = take_frame("cam")          # frame sent since the last run (or None)
    ... check the face, decide a message ...
    live_camera("cam", active=True, message="...", status="ok", ack=seq)
"""

import base64

import streamlit as st

from assets import face_engine
from assets.auto_capture import FRAME_INTERVAL_MS

HTML = """
<div class="cam" data-status="searching">
  <video autoplay playsinline muted></video>
  <div class="msg">Starting camera...</div>
</div>
"""

CSS = """
.cam {
  position: relative;
  max-width: 480px;
  border: 4px solid #999999;
  border-radius: 12px;
  overflow: hidden;
  background: #000000;
}
.cam[data-status="ok"]      { border-color: #0ca30c; }
.cam[data-status="error"]   { border-color: #d03b3b; }
.cam[data-status="paused"]  { border-color: #fab219; }
video {
  display: block;
  width: 100%;
  transform: scaleX(-1);          /* mirror, like a selfie camera */
}
.msg {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 8px 12px;
  background: rgba(0, 0, 0, 0.65);
  color: #ffffff;
  font: 600 15px/1.4 sans-serif;
  text-align: center;
}
"""

JS = """
// One entry per camera (keyed by cam_id), kept between Streamlit reruns.
const cams = new Map()

function startCamera(cam) {
  cam.starting = true
  navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
    .then((stream) => {
      cam.stream = stream
      cam.error = ""
      attachVideo(cam)
    })
    .catch(() => {
      cam.error = "Camera blocked. Allow camera access in the browser, or switch to Manual."
      if (cam.ui) cam.ui.msg.textContent = cam.error
    })
}

function attachVideo(cam) {
  const video = cam.ui && cam.ui.video
  if (video && cam.stream && video.srcObject !== cam.stream) {
    video.srcObject = cam.stream
    video.play().catch(() => {})
  }
}

function stopCamera(id) {
  const cam = cams.get(id)
  if (!cam) return
  clearInterval(cam.timer)
  if (cam.stream) cam.stream.getTracks().forEach((t) => t.stop())
  cams.delete(id)
}

// Runs 10x per second; sends a frame when it is time.
function tick(cam) {
  const { data, video, setTriggerValue } = cam.ui
  if (!data.active || !cam.stream || video.readyState < 2) return

  const now = Date.now()
  if (cam.waitingFor && now - cam.sentAt < 5000) return          // Python still busy
  if (now - cam.sentAt < (data.interval_ms || 400)) return       // not time yet

  const w = 640
  const h = Math.round(video.videoHeight * w / video.videoWidth)
  cam.canvas.width = w
  cam.canvas.height = h
  cam.canvas.getContext("2d").drawImage(video, 0, 0, w, h)

  cam.seq += 1
  cam.waitingFor = cam.seq
  cam.sentAt = now
  setTriggerValue("frame", { seq: cam.seq, image: cam.canvas.toDataURL("image/jpeg", 0.8) })
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const id = data.cam_id

  let cam = cams.get(id)
  if (!cam) {
    cam = { stream: null, starting: false, error: "", seq: 0, waitingFor: 0, sentAt: 0,
            timer: null, stopTimer: null, canvas: document.createElement("canvas") }
    cams.set(id, cam)
  }

  // a rerun is not an unmount: cancel the pending stop
  if (cam.stopTimer) {
    clearTimeout(cam.stopTimer)
    cam.stopTimer = null
  }

  const box = parentElement.querySelector(".cam")
  const ui = { data, setTriggerValue, box,
               video: parentElement.querySelector("video"),
               msg: parentElement.querySelector(".msg") }
  cam.ui = ui

  // show what Python says
  box.dataset.status = data.status || "searching"
  ui.msg.textContent = cam.error || data.message || ""

  // Python handled our last frame -> we may send the next one
  if (data.ack === cam.waitingFor) cam.waitingFor = 0

  if (!cam.stream && !cam.starting) startCamera(cam)
  attachVideo(cam)
  if (!cam.timer) cam.timer = setInterval(() => tick(cam), 100)

  // Called when Streamlit removes or re-renders the component.
  // Wait a moment: if it was only a rerun, the stop is cancelled above.
  return () => {
    cam.stopTimer = setTimeout(() => stopCamera(id), 1500)
  }
}
"""

_LIVE_CAMERA = st.components.v2.component("live_camera", html=HTML, css=CSS, js=JS)


def _remember_frame(key):
    # runs before the script when a new frame arrives; keep it for take_frame().
    # A late frame can arrive after the camera left the page (other tab, logout):
    # then the key or its frame is missing -> do nothing.
    value = st.session_state.get(key)
    frame = getattr(value, "frame", None) if value is not None else None

    if frame is None:
        return

    st.session_state[f"{key}__frame"] = frame


def take_frame(key):
    """The frame sent since the last run as (seq, BGR image), or None (also for a broken frame)."""

    frame = st.session_state.pop(f"{key}__frame", None)

    if not isinstance(frame, dict) or "seq" not in frame or "image" not in frame:
        return None

    try:
        image_b64 = frame["image"].split(",", 1)[1]      # drop "data:image/jpeg;base64,"
        image = face_engine.decode_image(base64.b64decode(image_b64))
    except Exception:
        return None

    if image is None:
        return None

    return frame["seq"], image


def live_camera(key, *, active, message, status, ack):
    """Show the live camera. status: "searching", "ok", "error" or "paused"."""

    _LIVE_CAMERA(
        key=key,
        data={
            "cam_id": key,
            "active": active,
            "message": message,
            "status": status,
            "ack": ack,
            "interval_ms": FRAME_INTERVAL_MS,
        },
        on_frame_change=lambda: _remember_frame(key),
    )
