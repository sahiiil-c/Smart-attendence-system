"""
assets/browser_location.py - takes 3 GPS readings in the visitor's browser.

Why our own component: the browser's GPS (navigator.geolocation) only runs in JavaScript.
This small custom component (Streamlit Components v2, like live_camera.py) asks for the
position 3 times, 2 seconds apart, and sends all 3 readings to Python together.

  * Python -> JS: data = {"request": 7, "count": 3, "gap_ms": 2000}
                  a NEW request number starts a new set of readings
  * JS -> Python: one trigger "readings":
        {"request": 7, "readings": [{"lat", "lon", "accuracy", "time"}, ...]}
     or {"request": 7, "error": "Location permission denied ..."}

Usage:
    result = take_readings("loc")          # readings sent since the last run (or None)
    request_location("loc", request_id)    # shows "Checking your location..." and starts it
"""

import streamlit as st

from assets.location_checks import READINGS_NEEDED, SECONDS_BETWEEN_READINGS

HTML = """
<div class="loc">📍 <span class="msg">Checking your location...</span></div>
"""

CSS = """
.loc {
  padding: 10px 14px;
  border: 2px solid #1f77b4;
  border-radius: 10px;
  font: 600 15px/1.4 sans-serif;
  max-width: 480px;
}
"""

JS = """
// remembers which request numbers were already started (kept between reruns)
const started = new Map()

function readOnce() {
  return new Promise((resolve, reject) => {
    navigator.geolocation.getCurrentPosition(
      (p) => resolve({ lat: p.coords.latitude, lon: p.coords.longitude,
                       accuracy: p.coords.accuracy, time: p.timestamp / 1000 }),
      (e) => reject(e),
      { enableHighAccuracy: true, maximumAge: 0, timeout: 15000 })   // fresh reading, no cache
  })
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms))

async function takeReadings(data, msg, setTriggerValue) {
  const readings = []
  try {
    for (let i = 1; i <= data.count; i++) {
      msg.textContent = "Checking your location... " + i + " / " + data.count
      readings.push(await readOnce())
      if (i < data.count) await wait(data.gap_ms)
    }
    msg.textContent = "Location checked."
    setTriggerValue("readings", { request: data.request, readings: readings })
  } catch (e) {
    const text = (e && e.code === 1)
      ? "Location permission denied. Allow location for this site and try again."
      : "Could not read your location (" + ((e && e.message) || "unknown error") + ")."
    msg.textContent = text
    setTriggerValue("readings", { request: data.request, error: text })
  }
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const msg = parentElement.querySelector(".msg")

  if (!navigator.geolocation) {
    msg.textContent = "This browser has no GPS / location support."
    return
  }

  // start each request only once, even if Streamlit re-renders the component
  if (data.request && started.get(data.loc_id) !== data.request) {
    started.set(data.loc_id, data.request)
    takeReadings(data, msg, setTriggerValue)
  }
}
"""

_LOCATION = st.components.v2.component("browser_location", html=HTML, css=CSS, js=JS)


def _remember_readings(key):
    # runs before the script when the readings arrive; keep them for take_readings().
    # A late answer after the component left the page is ignored (no KeyError).
    value = st.session_state.get(key)
    readings = getattr(value, "readings", None) if value is not None else None

    if readings is None:
        return

    st.session_state[f"{key}__readings"] = readings


def take_readings(key):
    """
    The answer sent since the last run, or None:
      {"request": 7, "readings": [...]}  or  {"request": 7, "error": "..."}
    """

    result = st.session_state.pop(f"{key}__readings", None)

    if not isinstance(result, dict) or "request" not in result:
        return None
    if "readings" not in result and "error" not in result:
        return None

    return result


def request_location(key, request_id):
    """Show the location box and take READINGS_NEEDED readings for this request number."""

    _LOCATION(
        key=key,
        data={
            "loc_id": key,
            "request": request_id,
            "count": READINGS_NEEDED,
            "gap_ms": SECONDS_BETWEEN_READINGS * 1000,
        },
        on_readings_change=lambda: _remember_readings(key),
    )
