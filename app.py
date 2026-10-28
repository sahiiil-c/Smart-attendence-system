"""
app4.py - AI Face Attendance (cloud version 4) = app2.py + location checks.

New in app4.py (see assets/location_checks.py for every check in plain English):
  * GEOFENCE        - students must be inside the college circle (centre + radius,
                      set by the admin in App settings).
  * MULTIPLE GPS    - 3 GPS readings 2 s apart; exactly identical readings or a round
    READINGS          accuracy (1/3/5/10 m) on every reading = suspicious (fake-GPS apps).
  * IMPOSSIBLE      - faster than 150 km/h since the student's last saved position = suspicious.
    TRAVEL
  * SAME SPOT       - another student on exactly the same point (6 decimals) in the last
                      10 minutes = both suspicious.
  * COLLEGE WI-FI   - the IP must be a college Wi-Fi address when WIFI_REQUIRED is on.
  GPS ON / OFF      - the admin can switch the GPS location check off (App settings); then
                      only the face and (if required) the college Wi-Fi are checked.
  No kiosk page     - everybody marks their OWN attendance after logging in.
  APPROVAL          - a new sign-up can't log in until the admin approves it
                      (students can also be approved by a teacher of their class).
  Blocked (nothing saved): outside the geofence, face not recognised, or Wi-Fi required
  and not on college Wi-Fi. Suspicious only: saved as "Flagged" for a teacher to
  Approve / Reject (Teacher -> Flagged attendance).

Everything below is the same as app2.py:

Differences from app.py:
  * Students and teachers SIGN UP (register their face) on the Login page.
  * After logging in they MARK THEIR OWN ATTENDANCE: the camera checks that the
    face really belongs to the logged-in person.
  * Students have a Year (FE/SE/TE/BE) and a BRANCH (app4.py: the admin's list in App settings,
    e.g. Computer, IT); teachers teach classes (year + branch, e.g. "SE-Computer") and see the
    students of their classes. The branch is stored in the "division" column.
  * Student and teacher attendance are kept in separate tables (see assets/storage.py).

Logging in:
  * Admin   - no button of its own: the admin username + password (from the secrets)
              typed into the Teacher login opens the admin dashboard
  * Teacher - username + password (chosen at sign-up, or created by the admin)
  * Student - Student ID, then the FACE is the password (camera check)
  You stay logged in after a page reload (cookie, see assets/auth.py) until Logout.

Pages (sidebar):
  1. Login            - Log in / Sign up; Teacher (or the admin), Student
  (app4.py has only this page: the shared kiosk "Mark Attendance" page was removed)

Run locally:  streamlit run app4.py
"""

import io
import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from assets import attendance_rules as rules
from assets import auth,auto_capture,browser_cookie,browser_location,face_engine,live_camera,storage
from assets import location_checks as checks
from assets.face_engine import MATCH_THRESHOLD
from assets.storage import YEARS, class_name, students_of_teacher

REGISTER_PHOTOS = 3      # photos taken per person at registration
LOGIN_COOKIE = "fa_login"  # browser cookie that keeps you logged in

# status colours (always shown together with their text label)
STATUS_COLORS = {"Present": "#0ca30c", "Pending": "#fab219", "Absent": "#d03b3b",
                 "Flagged": "#8a4bd6"}   # Flagged = suspicious, waiting for a teacher

st.set_page_config(page_title="AI Attendance Management", page_icon="logo/streamlit_page_icon.png", layout="wide")

# The login box (see page_login) is at least as tall as the screen, so its content can sit
# in the vertical middle. Longer content (the sign-up camera) simply makes the box taller.
st.html("<style>.st-key-login_box { min-height: calc(100vh - 8rem); }</style>")


# ---------------- LOADED ONCE ---------------- #

@st.cache_resource(show_spinner="Loading the face model... (the first start can take a few minutes)")
def get_model():
    return face_engine.load_model()


@st.cache_resource
def get_store(mode):
    return storage.get_store(mode)


# the admin chooses offline (SQLite) or online (Supabase) for the whole app
store = get_store(storage.get_mode())


# ---------------- SMALL HELPERS ---------------- #

def format_name(text):
    """'sahil g chavan' -> 'Sahil G Chavan' (same rule as the old project)."""

    parts = text.strip().split()

    if len(parts) < 2:
        return None

    first, last = parts[0].capitalize(), parts[-1].capitalize()

    if len(parts) > 2:
        return f"{first} {parts[1][0].upper()} {last}"

    return f"{first} {last}"


def capture_mode(key):
    """Auto (live camera, no clicking) or Manual (Take photo button)."""

    # the admin can hide the switch: then everyone uses the default mode
    if not storage.get_capture_switch_visible():
        return storage.get_capture_default()

    return st.segmented_control(
        "Capture mode", ["Auto", "Manual"], default=storage.get_capture_default(),
        required=True, key=key,
        help="Auto takes photos by itself when it sees a face. "
             "Use Manual if the live camera does not work on this device."
    )


def recognise(emb):
    """embedding -> (person_id or None, score)"""

    ids, embs = store.get_all_embeddings()
    person_id, score = face_engine.best_match(emb, ids, embs)

    if person_id is None or score < MATCH_THRESHOLD:
        return None, score

    return person_id, score


def valid_time(text):
    """'' or 'HH:MM:SS' -> True"""

    if not text:
        return True

    try:
        datetime.strptime(text, "%H:%M:%S")
        return True
    except ValueError:
        return False


def approval_state(p):
    """
    app4.py "approved" column:
       1 (or empty) -> "approved"  may log in
       0            -> "pending"   new sign-up waiting for approval
      -1            -> "rejected"  login rejected: data kept, the login says "Invalid ID"
    """
    return {0: "pending", -1: "rejected"}.get(p.get("approved"), "approved")


def is_pending(p):
    return approval_state(p) == "pending"


def is_rejected(p):
    return approval_state(p) == "rejected"


def full_report():
    # people waiting for approval or with a rejected login are not in the reports
    people = [p for p in store.get_people() if approval_state(p) == "approved"]
    return rules.build_report(people, store.get_attendance(), rules.now_local())


def class_students(teacher_id, state="approved"):
    """Students of the teacher's classes with this approval state (approved / pending / rejected)."""
    return [s for s in students_of_teacher(store, teacher_id) if approval_state(s) == state]


def this_month(df):
    month = rules.now_local().strftime("%Y-%m")
    return df[df["Date"].str.startswith(month)]


def count_status(df):
    """Present / Pending / Absent / Flagged counts (app4 also shows Flagged)."""
    return rules.count_status(df, with_flagged=True)


def filter_dates(df, start, end):
    """Rows from start to end (both included). Dates are 'YYYY-MM-DD' text, so text compare works."""
    return df[(df["Date"] >= start.isoformat()) & (df["Date"] <= end.isoformat())]


def show_status_metrics(counts):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Present", counts["Present"])
    c2.metric("🟡 Pending", counts["Pending"])
    c3.metric("🔴 Absent", counts["Absent"])
    c4.metric("🟣 Flagged", counts.get("Flagged", 0))


def status_bar_chart(counts, title):
    """Bar chart of Present / Pending / Absent. Never drawn when everything is 0."""

    if sum(counts.values()) == 0:
        st.info("No attendance records yet")
        return

    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, values, width=0.5,
                  color=[STATUS_COLORS[l] for l in labels])

    ax.bar_label(bars, padding=3, color="#333333")   # value on each bar
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel("Days")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    st.pyplot(fig)
    plt.close(fig)


def per_person_chart(df):
    """Stacked horizontal bars: Present / Pending / Absent days per person."""

    if df.empty:
        st.info("No attendance records yet")
        return

    table = (df.groupby("Name")["Attendance"].value_counts().unstack(fill_value=0)
             .reindex(columns=list(STATUS_COLORS), fill_value=0))

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(table) + 1.2))
    left = np.zeros(len(table))

    for status, color in STATUS_COLORS.items():
        ax.barh(table.index, table[status], left=left, height=0.5,
                color=color, edgecolor="white", linewidth=1.5, label=status)
        left += table[status].values

    ax.set_xlabel("Days")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncols=4, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    st.pyplot(fig)
    plt.close(fig)


def excel_bytes(df):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, sheet_name="Attendance")
    return buffer.getvalue()


def person_label(p):
    group = f", {class_name(p['year'], p['division'])}" if p.get("year") else ""
    frozen = " 🔒 frozen" if p.get("frozen") else ""
    pending = (" ⏳ pending" if is_pending(p) else
               " ⛔ login rejected" if is_rejected(p) else "")
    return f"{p['id']} – {p['name']} ({p['role']}{group}){frozen}{pending}"


def all_classes(extra=()):
    """
    Every Year x Branch: [('FE', 'Computer'), ('FE', 'IT'), ...].
    extra = classes someone already has that are no longer in the list (kept, never lost).
    """
    classes = [(y, b) for y in YEARS for b in storage.get_branches()]
    return classes + [c for c in extra if c not in classes]


def branch_label(branch):
    """A branch that was removed from the list (or an old division A-D) is shown as '(old)'."""
    if branch in ("–", "All") or branch in storage.get_branches():
        return branch
    return f"{branch} (old)"


def class_label(c):
    """('SE', 'Computer') -> 'SE-Computer', or 'SE-A (old)' for an old value."""
    return class_name(*c) + ("" if c[1] in storage.get_branches() else " (old)")


# ======================================================================
# SIGN UP (register face) - shown on the Login page
# ======================================================================

def reset_registration():
    ss = st.session_state
    ss.reg_embs = []
    ss.reg_last = 0.0
    ss.reg_cam_key = ss.get("reg_cam_key", 0) + 1
    ss.reg_cam_msg, ss.reg_cam_status = "Look at the camera", "searching"
    ss.id_checked = False                      # a teacher must pass the identity check again


# ---- teacher sign-up: identity check BEFORE any photo is taken ----

IDENTITY_SCANS = 3        # face scans a new teacher must pass


def reset_identity_check():
    ss = st.session_state
    ss.id_scans = 0
    ss.id_cam = ("Look at the camera", "searching")
    ss.id_cam_key = ss.get("id_cam_key", 0) + 1


def cancel_teacher_signup(student):
    """
    A registered STUDENT's face was seen in a teacher sign-up:
    freeze the student, log them out everywhere, tell the admin,
    cancel the sign-up and go back to the Log in screen.
    """

    ss = st.session_state
    now = rules.now_local().strftime("%Y-%m-%d %H:%M:%S")
    details = ss.get("signup_details") or {}
    name = (details.get("name") or "").strip() or "?"
    username = (details.get("username") or "").strip() or "?"

    store.set_frozen(student["id"], True)
    store.delete_login_sessions_of(student["id"])
    store.add_alert(student["id"],
                    f"Tried to sign up as teacher '{name}' (username '{username}')", now)

    cancel_signup_capture()
    ss.go_to_login = True
    st.rerun(scope="app")


def identity_scan(emb):
    """
    One identity-check scan of a new teacher.
    Returns "scanning" (clean, more scans needed), "passed" (3 clean scans) or
    "blocked" (face of a registered teacher). A student face never returns: it cancels.
    """

    ss = st.session_state
    person_id, score = recognise(emb)

    if person_id:
        person = store.get_person(person_id)

        if person["role"] == "student":
            cancel_teacher_signup(person)          # even ONE scan is enough

        ss.reg_error = (f"⚠ This face is already registered as {person['name']} "
                        f"({person['role']}, {person['id']}). Registration blocked.")
        reset_identity_check()
        return "blocked"

    ss.id_scans += 1

    if ss.id_scans >= IDENTITY_SCANS:
        ss.id_checked = True
        return "passed"

    return "scanning"


@st.fragment
def identity_check_auto():
    """Live camera: every frame with one face is one scan."""

    ss = st.session_state
    if "id_scans" not in ss:
        reset_identity_check()

    frame = live_camera.take_frame("idcheck_live")

    if frame:
        seq, img = frame
        ss.id_ack = seq

        face, message = face_engine.get_face(get_model(), img)

        if face is None:
            ss.id_cam = (message, "searching")
        else:
            if identity_scan(face.normed_embedding) in ("passed", "blocked"):
                st.rerun(scope="app")
            ss.id_cam = (f"🛡 Scan {ss.id_scans} / {IDENTITY_SCANS} – hold still", "ok")

    st.progress(ss.id_scans / IDENTITY_SCANS,
                text=f"Identity check: {ss.id_scans} / {IDENTITY_SCANS} scans")

    message, status = ss.id_cam
    live_camera.live_camera("idcheck_live", active=True, message=message, status=status,
                            ack=ss.get("id_ack", 0))


def identity_check_manual():
    """Each photo is one scan."""

    ss = st.session_state
    if "id_scans" not in ss:
        reset_identity_check()

    photo = st.camera_input(f"Identity scan {ss.id_scans + 1} / {IDENTITY_SCANS}",
                            key=f"id_cam_{ss.id_cam_key}")

    if photo is None:
        return

    with st.spinner("Checking face..."):
        img = face_engine.decode_image(photo.getvalue())
        emb, message = face_engine.get_face_embedding(get_model(), img)

    if emb is None:
        st.error(f"{message} Click **Clear photo** and try again.")
        return

    if identity_scan(emb) == "scanning":
        ss.id_cam_key += 1                     # fresh camera for the next scan
    st.rerun()


def add_registration_photo(emb):
    """
    Used by Auto AND Manual mode.
    Returns (ok, message). On a duplicate face the registration is reset.
    """

    ss = st.session_state
    ids, embs = store.get_all_embeddings()
    ok, message, existing_id = auto_capture.check_registration_photo(emb, ss.reg_embs, ids, embs)

    if existing_id:
        person = store.get_person(existing_id)

        # a student's face in a TEACHER sign-up: cancel, freeze, alert the admin
        if person["role"] == "student" and ss.get("signup_role") == "teacher":
            cancel_teacher_signup(person)

        message = (f"⚠ This face is already registered as {person['name']} "
                   f"({person['role']}, {person['id']}). Registration blocked.")
        reset_registration()

    if ok:
        ss.reg_embs.append(emb)
        ss.reg_last = time.time()

    return ok, message


@st.fragment
def register_auto():
    """Live camera: takes the photos by itself. Only this part reruns per frame."""

    ss = st.session_state
    if "reg_embs" not in ss:                   # state missing (e.g. a late frame): start fresh
        reset_registration()

    frame = live_camera.take_frame("reg_live")

    if frame:
        seq, img = frame
        ss.reg_ack = seq

        face, message = face_engine.get_face(get_model(), img)
        status = "searching"

        if face is not None:
            message = face_engine.registration_checks(face, img)
            status = "error" if message else "ok"

        if face is not None and message is None:
            emb = face.normed_embedding
            message = auto_capture.auto_registration_ready(emb, ss.reg_embs, ss.reg_last, time.time())

            if message is None:
                ok, message = add_registration_photo(emb)

                if not ok:
                    ss.reg_error = message
                    st.rerun(scope="app")       # show the error on the full page

                if len(ss.reg_embs) == REGISTER_PHOTOS:
                    st.rerun(scope="app")       # all photos taken -> show Save button

                message = f"📸 {len(ss.reg_embs)} / {REGISTER_PHOTOS} – now turn your head slightly"

        ss.reg_cam_msg, ss.reg_cam_status = message, status

    taken = len(ss.reg_embs)
    st.progress(taken / REGISTER_PHOTOS, text=f"Photos taken: {taken} / {REGISTER_PHOTOS}")

    live_camera.live_camera("reg_live", active=True, message=ss.reg_cam_msg,
                            status=ss.reg_cam_status, ack=ss.get("reg_ack", 0))


def register_manual():
    """Old way: click 'Take photo' for each photo."""

    ss = st.session_state
    if "reg_embs" not in ss:
        reset_registration()

    taken = len(ss.reg_embs)
    st.progress(taken / REGISTER_PHOTOS, text=f"Photos taken: {taken} / {REGISTER_PHOTOS}")

    # a new key after each accepted photo gives a fresh, empty camera
    photo = st.camera_input(f"Photo {taken + 1}", key=f"reg_cam_{ss.reg_cam_key}")

    if photo is None:
        return

    with st.spinner("Checking face..."):
        img = face_engine.decode_image(photo.getvalue())
        emb, message = face_engine.get_face_embedding(get_model(), img)

    if emb is None:
        st.error(f"{message} Click **Clear photo** and try again.")
        return

    ok, message = add_registration_photo(emb)

    if not ok:
        st.error(message)
        return

    ss.reg_cam_key += 1
    st.rerun()


def teacher_login_problem(username, password, confirm):
    """What is wrong with a new teacher's username / password, or None."""

    if not username.strip() or not password:
        return "Choose a username and a password to continue."
    if store.get_teacher_account(username) or auth.is_admin_username(username):
        return "This username is already taken."
    if len(password) < 6:
        return "The password needs at least 6 characters."
    if password != confirm:
        return "The two passwords are not the same."
    return None


def cancel_signup_capture():
    """Back to the details form: forget the submitted details, the scans and the photos."""

    ss = st.session_state
    ss.pop("signup_details", None)
    reset_registration()
    reset_identity_check()


def signup_form():
    """
    Step 1 of sign-up: the details. Nothing is filmed yet - the camera starts
    only after "Submit and start camera" (and only if the details are OK).
    """

    ss = st.session_state

    role = st.segmented_control("I am a", ["student", "teacher"], default="student",
                                required=True, format_func=str.capitalize, key="signup_role")

    branches = storage.get_branches()
    if role == "student" and not branches:
        st.warning("No branches yet – please ask the admin to add them.")
        return

    with st.form("signup_form"):
        name_input = st.text_input("Full Name (first and last name)", key="signup_name")

        if role == "student":
            c1, c2 = st.columns(2)
            year = c1.selectbox("Year", YEARS, key="signup_year")
            division = c2.selectbox("Branch", branches, key="signup_div")   # stored as "division"
            classes, username, password, confirm = [], "", "", ""
        else:
            year = division = None
            classes = st.multiselect("Classes I teach (Year-Branch; the admin can change this later)",
                                     all_classes(), format_func=lambda c: class_name(*c),
                                     key="signup_classes")

            # teachers log in with a username + password
            c1, c2, c3 = st.columns(3)
            username = c1.text_input("Username", key="signup_user")
            password = c2.text_input("Password (min. 6 characters)", type="password",
                                     key="signup_pass")
            confirm = c3.text_input("Confirm password", type="password", key="signup_pass2")

        submitted = st.form_submit_button("✅ Submit and start camera", type="primary")

    st.caption("The camera starts after you press **Submit**.")

    if not submitted:
        return

    # check everything BEFORE the camera starts, so nobody takes photos for nothing
    fullname = format_name(name_input)
    if not fullname:
        st.error("Enter at least a first and last name.")
        return

    if role == "teacher":
        problem = teacher_login_problem(username, password, confirm)
        if problem:
            st.error(problem)
            return

    ss.signup_details = {"role": role, "name": fullname, "year": year, "division": division,
                         "classes": classes, "username": username, "password": password}
    reset_registration()
    reset_identity_check()
    st.rerun()


def signup():
    """
    Sign up = register your face, in 2 steps:
      1. fill in the details and press Submit  (signup_form - nothing is filmed yet)
      2. the camera starts: identity check (teachers) -> 3 photos -> Save sign-up
    """

    ss = st.session_state
    if "reg_embs" not in ss:
        reset_registration()

    if ss.get("reg_error"):
        st.error(ss.pop("reg_error"))

    details = ss.get("signup_details")

    if not details:
        signup_form()
        return

    # ---- step 2: the camera, with the SUBMITTED details ----
    role, fullname = details["role"], details["name"]
    year, division, classes = details["year"], details["division"], details["classes"]
    username, password = details["username"], details["password"]

    col1, col2 = st.columns([8, 2], vertical_alignment="center")
    about = (f"{class_name(year, division)}" if role == "student"
             else f"username **{username.strip().lower()}**")
    col1.info(f"**{fullname}** – {role}, {about}")
    if col2.button("✏️ Edit details"):
        cancel_signup_capture()
        st.rerun()

    # teachers: identity check first - is this face already registered as a student?
    if role == "teacher" and not ss.get("id_checked"):
        if "id_scans" not in ss:
            reset_identity_check()

        st.info(f"🛡 **Identity check** – we scan your face {IDENTITY_SCANS} times before "
                "registering you as a teacher. Photos are taken after that.")

        if capture_mode("id_mode") == "Auto":
            identity_check_auto()
        else:
            identity_check_manual()
        return

    if len(ss.reg_embs) < REGISTER_PHOTOS:

        mode = capture_mode("reg_mode")

        st.info(f"We take {REGISTER_PHOTOS} photos. Between photos, **turn your head slightly** "
                "or move a little.")

        if mode == "Auto":
            register_auto()
        else:
            register_manual()
        return

    # ---- all photos taken ----
    st.success(f"All {REGISTER_PHOTOS} photos taken for **{fullname}** ({role}).")

    col1, col2 = st.columns(2)

    if col1.button("✅ Save sign-up", type="primary"):
        new_id = store.next_person_id(role)
        today = rules.now_local().date().isoformat()

        store.save_person(new_id, fullname, role, today, year, division)
        store.set_approved(new_id, False)          # must be approved before logging in
        store.save_embeddings(new_id, ss.reg_embs)

        if role == "teacher":
            store.set_teacher_classes(new_id, classes)
            store.save_teacher_account(username, auth.hash_password(password), new_id)
            how = f"Log in next time with the username **{username.strip().lower()}**."
        else:
            how = (f"Log in next time with the number **{new_id.removeprefix('ID_')}** "
                   "and your face.")

        who = "the admin" if role == "teacher" else "the admin or a teacher of your class"
        cancel_signup_capture()                    # also forgets the password in the session
        ss.signup_done = (f"✅ Signed up, {fullname}! Your ID is **{new_id}**. "
                          f"You can log in after {who} approves your sign-up. {how}")
        st.rerun()

    if col2.button("Start over"):
        cancel_signup_capture()
        st.rerun()


# ======================================================================
# MARKING ATTENDANCE (Time In / Time Out rules)
# ======================================================================

def mark_attendance(person_id):
    """Apply the Time In / Time Out rules. Returns (kind, message) to show."""

    person = store.get_person(person_id)
    name = person["name"]

    if person.get("frozen"):
        return "error", f"🔒 {name}'s account is frozen – please contact the admin."

    now = rules.now_local()
    today = now.date().isoformat()
    clock = now.strftime("%H:%M:%S")

    # students and teachers have their own cooldown (Admin -> App settings)
    cooldown = storage.get_cooldown(person["role"])
    action, seconds_left = rules.decide_scan(store.get_record(person_id, today), now, cooldown)

    if action == "time_in":
        store.mark_time_in(person_id, today, clock)
        return "success", f"✅ {name} — Time In marked at {clock}"

    if action == "time_out":
        store.mark_time_out(person_id, today, clock)
        return "success", f"✅ {name} — Time Out marked at {clock}"

    if action == "cooldown":
        return "warning", (f"⏳ {name}, please wait {rules.wait_text(seconds_left)} "
                           "before marking Time Out.")

    return "info", f"{name}, your attendance for today is already complete."


# ======================================================================
# LOCATION + WI-FI CHECKS (anti-fake-GPS, the rules are in assets/location_checks.py)
# ======================================================================

def college_settings():
    """
    The college geofence and Wi-Fi settings.
    Starting values come from st.secrets; what the admin saves in App settings
    (settings.json) wins, because an app can't change its own secrets.
    """

    def pick(name, secret, default):
        value = storage.get_setting(name, None)
        return value if value is not None else storage.get_secret(secret, default)

    allowed = pick("allowed_ips", "ALLOWED_IPS", [])
    if isinstance(allowed, str):                    # "1.2.3.4, 10.0.0.0/24" in the secrets
        allowed = allowed.replace("\n", ",").split(",")

    lat = pick("college_lat", "COLLEGE_LAT", None)
    lon = pick("college_lon", "COLLEGE_LON", None)

    return {
        "lat": float(lat) if lat not in (None, "") else None,
        "lon": float(lon) if lon not in (None, "") else None,
        "radius_m": float(pick("geofence_radius_m", "GEOFENCE_RADIUS_M", 200)),
        "allowed_ips": [str(a).strip() for a in allowed if str(a).strip()],
        "wifi_required": bool(pick("wifi_required", "WIFI_REQUIRED", False)),
        "gps_required": bool(pick("gps_required", "GPS_REQUIRED", True)),   # GPS check on/off
    }


def gps_on():
    """True when the admin has the GPS location check switched on (default)."""
    return college_settings()["gps_required"]


def current_ip():
    """The visitor's IP: CF-Connecting-IP, then X-Forwarded-For, then the connection itself."""

    try:
        headers = dict(st.context.headers or {})
        fallback = st.context.ip_address or "127.0.0.1"      # None on localhost
    except Exception:
        headers, fallback = {}, "127.0.0.1"

    return checks.client_ip(headers, fallback)


def clean_readings(readings):
    """Keep only complete, numeric readings - never trust what a browser sends."""

    good = []
    for r in readings or []:
        try:
            good.append({k: float(r[k]) for k in ("lat", "lon", "accuracy", "time")})
        except (KeyError, TypeError, ValueError):
            return []

    return good if len(good) == checks.READINGS_NEEDED else []


def reset_location(key):
    st.session_state.pop(f"{key}_state", None)


def location_step(key):
    """
    Takes the GPS readings once (3 readings, 2 s apart, in the browser).
    Returns the readings when they are ready, otherwise None
    (still reading, or an error is shown with a Try again button).
    """

    ss = st.session_state
    state = dict(ss.get(f"{key}_state") or {})

    answer = browser_location.take_readings(key)
    if answer and answer.get("request") == state.get("request"):
        state["answer"] = answer
        ss[f"{key}_state"] = state

    answer = state.get("answer")

    if answer:
        readings = clean_readings(answer.get("readings"))
        if readings:
            return readings

        st.error(answer.get("error") or "The location could not be read.")
        if st.button("🔁 Try again", key=f"{key}_retry"):
            reset_location(key)
            st.rerun()
        return None

    if not state.get("request"):
        state["request"] = time.time_ns()           # a new number = a new set of readings
        ss[f"{key}_state"] = state

    browser_location.request_location(key, state["request"])
    return None


def saved_positions():
    """
    Every saved student position, from the gps_readings column:
    [{"person_id", "lat", "lon", "time"}] - time = when it was saved (server clock).
    """

    positions = []

    for r in store.get_attendance("student"):
        try:
            saved = json.loads(r.get("gps_readings") or "{}")
        except ValueError:
            continue

        readings = clean_readings(saved.get("readings")) if isinstance(saved, dict) else []
        if readings and saved.get("saved_at"):
            lat, lon = checks.average_position(readings)
            positions.append({"person_id": r["person_id"], "lat": lat, "lon": lon,
                              "time": float(saved["saved_at"])})

    return positions


def check_location(person_id, readings, gps_checks=True):
    """
    Geofence + college Wi-Fi, and (for one student) the suspicious-GPS checks.
    The face is checked separately: nothing is saved unless the face matches.
    readings = [] (GPS switched off) -> only the college Wi-Fi rule.
    Returns {"kind": "blocked" / "flagged" / "ok", "detail", "flags", "others", "ip", "readings"}.
    """

    cfg = college_settings()
    ip = current_ip()
    result = {"ip": ip, "readings": readings or [], "flags": [], "others": []}
    ip_ok = checks.ip_allowed(ip, cfg["allowed_ips"])

    # GPS check switched OFF by the admin (no readings): only the Wi-Fi rule is left
    if not readings:
        kind, detail = checks.decide(True, True, cfg["wifi_required"], ip_ok, [])
        return {**result, "kind": kind, "detail": detail}

    if cfg["lat"] is None or cfg["lon"] is None:
        return {**result, "kind": "blocked",
                "detail": "The college location is not set yet – please ask the admin."}

    lat, lon = checks.average_position(readings)          # the AVERAGE is used
    inside = checks.inside_geofence(lat, lon, cfg["lat"], cfg["lon"], cfg["radius_m"])

    flags, others = [], []
    if gps_checks:
        positions = saved_positions()
        mine = [p for p in positions if p["person_id"] == person_id]
        last = max(mine, key=lambda p: p["time"]) if mine else None
        flags, others = checks.gps_flags(readings, last, positions, person_id, time.time())

    kind, detail = checks.decide(inside, True, cfg["wifi_required"], ip_ok, flags)
    return {**result, "kind": kind, "detail": detail, "flags": flags, "others": others}


def show_location_result(verdict):
    if verdict["kind"] == "blocked":
        st.error(f"⛔ {verdict['detail']}")
    elif verdict["kind"] == "flagged":
        # the reason is not shown to the student, only to the teachers
        st.warning("📍 Location checked – something looks unusual. "
                   "A teacher will review this attendance.")
    elif verdict.get("readings"):
        st.success("📍 Location checked – you are at college.")
    elif college_settings()["wifi_required"]:
        st.success("📶 Connected to the college Wi-Fi.")
    # GPS and Wi-Fi checks both off: nothing to show


def save_evidence(person_id, verdict, extra_flags=()):
    """
    After Time In / Time Out was saved: store the IP and the 3 readings on today's record,
    plus the flag and its reason. Students seen on the same spot are flagged too.
    Returns the list of reasons (empty = not flagged).
    """

    today = rules.now_local().date().isoformat()
    record = store.get_record(person_id, today) or {}

    reasons = set(verdict.get("flags") or []) | set(extra_flags or [])
    if record.get("flagged") and record.get("flag_reason"):         # keep an earlier flag of today
        reasons |= {r.strip() for r in record["flag_reason"].split(";") if r.strip()}

    others = verdict.get("others") or []
    if others:
        reasons.discard(checks.FLAG_SAME_LOCATION)
        reasons.add(f"{checks.FLAG_SAME_LOCATION} (with {', '.join(others)})")

    gps_json = json.dumps({"saved_at": time.time(), "readings": verdict["readings"]})
    store.save_check_info(person_id, today, verdict["ip"], gps_json,
                          bool(reasons), "; ".join(sorted(reasons)))

    # mark BOTH students: the other one's record of today gets the same flag
    for other in others:
        rec = store.get_record(other, today)
        if rec:
            theirs = {r.strip() for r in (rec.get("flag_reason") or "").split(";") if r.strip()}
            theirs.add(f"{checks.FLAG_SAME_LOCATION} (with {person_id})")
            store.save_check_info(other, today, rec.get("ip_address"), rec.get("gps_readings"),
                                  True, "; ".join(sorted(theirs)))

    return sorted(reasons)


def mark_my_record(user):
    """Mark attendance for the logged-in user; students also get the location evidence saved."""

    kind, text = mark_attendance(user["id"])

    if user["role"] != "Student" or kind != "success":
        return kind, text

    verdict = st.session_state.get("me_verdict")
    if not verdict:
        return kind, text

    reasons = save_evidence(user["id"], verdict, st.session_state.get("login_flags") or [])
    if reasons:
        return "warning", f"{text} – 🚩 flagged for teacher review"

    return kind, text


# ======================================================================
# MARK MY ATTENDANCE (logged-in student / teacher)
# ======================================================================

def is_me(emb, person_id):
    """True when the face matches this person's OWN saved photos."""

    ids, embs = store.get_all_embeddings()
    mine = embs[ids == person_id]

    return len(mine) > 0 and float(np.max(mine @ emb)) >= MATCH_THRESHOLD


def reset_my_attendance():
    ss = st.session_state
    ss.pop("me_verdict", None)                 # a new mark = a new location check
    reset_location("me_loc")
    ss.me_result = None
    ss.me_watch = auto_capture.new_watch_state()
    ss.me_cam = ("Look at the camera", "searching")
    ss.me_cam_key = ss.get("me_cam_key", 0) + 1


@st.fragment
def my_attendance_auto(user):
    """Live camera: the same person must be seen in 2 frames in a row, then it marks."""

    ss = st.session_state
    if not ss.get("user"):                     # logged out meanwhile: ignore late frames
        ss.pop("me_live__frame", None)
        return
    if "me_watch" not in ss:
        reset_my_attendance()

    frame = live_camera.take_frame("me_live")

    if frame:
        seq, img = frame
        ss.me_ack = seq

        face, message = face_engine.get_face(get_model(), img)
        person_id = None

        if face is not None and is_me(face.normed_embedding, user["id"]):
            person_id = user["id"]

        step = auto_capture.attendance_step(ss.me_watch, person_id, time.time())

        if face is None:
            ss.me_cam = (message, "searching")
        elif step == "unknown":
            ss.me_cam = (f"This face is not {user['name']}", "error")
        elif step == "confirming":
            ss.me_cam = ("Hold still...", "ok")
        else:                                       # "mark"
            ss.me_result = mark_my_record(user)
            st.rerun(scope="app")                   # show the result, camera closes

    message, status = ss.me_cam
    live_camera.live_camera("me_live", active=True, message=message, status=status,
                            ack=ss.get("me_ack", 0))


def my_attendance_manual(user):
    """Click 'Take photo'. The photo must be of the logged-in person."""

    ss = st.session_state
    if not ss.get("user"):
        return
    if "me_watch" not in ss:
        reset_my_attendance()

    photo = st.camera_input("Take a photo of your face", key=f"me_cam_{ss.me_cam_key}")

    if photo is None:
        return

    with st.spinner("Checking face..."):
        img = face_engine.decode_image(photo.getvalue())
        emb, message = face_engine.get_face_embedding(get_model(), img)

    if emb is None:
        st.error(f"{message} Click **Clear photo** and try again.")
    elif not is_me(emb, user["id"]):
        st.error(f"This face is not {user['name']}. Nothing was saved.")
    else:
        ss.me_result = mark_my_record(user)
        st.rerun()


def mark_my_attendance(user):

    ss = st.session_state
    if "me_watch" not in ss:
        reset_my_attendance()

    if ss.me_result:
        kind, text = ss.me_result
        getattr(st, kind)(f"### {text}")      # st.success / st.warning / st.info

        if st.button("📷 Scan again"):
            reset_my_attendance()
            st.rerun()
        return

    # students: the LOCATION is checked first; the camera only opens if it passes
    if user["role"] == "Student":
        readings = location_step("me_loc") if gps_on() else []    # [] = GPS check off
        if readings is None:
            return
        if not ss.get("me_verdict"):
            ss.me_verdict = check_location(user["id"], readings)

        show_location_result(ss.me_verdict)

        if ss.me_verdict["kind"] == "blocked":
            if st.button("🔁 Check again", key="me_loc_again"):
                reset_my_attendance()
                st.rerun()
            return

    st.caption("Look at the camera. Attendance is marked only if the face is yours.")

    if capture_mode("me_mode") == "Auto":
        my_attendance_auto(user)
    else:
        my_attendance_manual(user)


# ======================================================================
# PAGE 1 - LOGIN (log in / sign up) + DASHBOARDS
# ======================================================================

def log_in(user):
    """Log in for this visit AND remember it in the browser (survives a page reload)."""

    ss = st.session_state
    ss.user = user
    ss.login_token = auth.start_login(store, user)

    # suspicious GPS flags seen at login are added to today's attendance later
    ss.login_flags = (ss.get("login_verdict") or {}).get("flags", [])

    # forget the half-finished student login (location + face)
    for key in [k for k in ss if str(k).startswith("login_face")]:
        del ss[key]
    ss.pop("login_verdict", None)
    reset_location("login_loc")
    ss.login_pending = None


def log_out():
    ss = st.session_state
    auth.end_login(store, ss.get("login_token"))
    ss.user = None
    ss.login_token = None
    ss.login_flags = []
    for key in [k for k in ss if str(k).startswith("me_")]:
        del ss[key]


# ---- student login: Student ID, then the face is the password ----

def reset_face_login():
    ss = st.session_state
    ss.login_face_watch = auto_capture.new_watch_state()
    ss.login_face_cam = ("Look at the camera", "searching")
    ss.login_face_key = ss.get("login_face_key", 0) + 1


@st.fragment
def face_login_auto(person):
    """Live camera: the student's own face in 2 frames in a row -> logged in."""

    ss = st.session_state
    if not ss.get("login_pending"):            # login finished or cancelled: ignore late frames
        ss.pop("login_live__frame", None)
        return
    if "login_face_watch" not in ss:
        reset_face_login()

    frame = live_camera.take_frame("login_live")

    if frame:
        seq, img = frame
        ss.login_face_ack = seq

        face, message = face_engine.get_face(get_model(), img)
        person_id = None

        if face is not None and is_me(face.normed_embedding, person["id"]):
            person_id = person["id"]

        step = auto_capture.attendance_step(ss.login_face_watch, person_id, time.time())

        if face is None:
            ss.login_face_cam = (message, "searching")
        elif step == "unknown":
            ss.login_face_cam = (f"This face is not {person['name']}", "error")
        elif step == "confirming":
            ss.login_face_cam = ("Hold still...", "ok")
        else:                                       # face confirmed
            log_in({"role": "Student", "id": person["id"], "name": person["name"]})
            st.rerun(scope="app")

    message, status = ss.login_face_cam
    live_camera.live_camera("login_live", active=True, message=message, status=status,
                            ack=ss.get("login_face_ack", 0))


def face_login_manual(person):
    """Click 'Take photo'. Logged in only if the photo is of this student."""

    ss = st.session_state
    if not ss.get("login_pending"):
        return
    if "login_face_watch" not in ss:
        reset_face_login()

    photo = st.camera_input("Take a photo of your face", key=f"login_face_photo_{ss.login_face_key}")

    if photo is None:
        return

    with st.spinner("Checking face..."):
        img = face_engine.decode_image(photo.getvalue())
        emb, message = face_engine.get_face_embedding(get_model(), img)

    if emb is None:
        st.error(f"{message} Click **Clear photo** and try again.")
    elif not is_me(emb, person["id"]):
        st.error(f"This face is not {person['name']}. Access denied.")
    else:
        log_in({"role": "Student", "id": person["id"], "name": person["name"]})
        st.rerun()


def student_face_login():

    ss = st.session_state
    person = ss.login_pending

    col1, col2 = st.columns([8, 2], vertical_alignment="center")
    col1.info(f"Hi **{person['name']}** ({person['id']}) – "
              + ("first we check your location, then " if gps_on() else "")
              + "we check your face (it is your password).")
    if col2.button("✖ Cancel"):
        ss.login_pending = None
        ss.pop("login_verdict", None)
        reset_location("login_loc")
        st.rerun()

    # 1. location: geofence + Wi-Fi (+ suspicious-GPS checks); blocked = no login.
    #    With the GPS check off only the Wi-Fi rule is checked here.
    readings = location_step("login_loc") if gps_on() else []   # [] = GPS check off
    if readings is None:
        return
    if not ss.get("login_verdict"):
        ss.login_verdict = check_location(person["id"], readings)

    show_location_result(ss.login_verdict)
    if ss.login_verdict["kind"] == "blocked":
        return

    # 2. face
    if capture_mode("login_face_mode") == "Auto":
        face_login_auto(person)
    else:
        face_login_manual(person)


def student_id_from(text):
    """
    What the student typed after "ID_" -> the full Student ID.
    '1', '01', '001' -> 'ID_001'  (3 digits, like the IDs are made)
    'id_001' typed anyway -> 'ID_001'.  Anything else -> None.
    """

    text = text.strip().upper().removeprefix("ID_")

    if not text.isdigit():
        return None

    return f"ID_{int(text):03}"


def page_login():

    ss = st.session_state
    user = ss.get("user")

    if user:
        st.success(f"Logged in as **{user['name']}** ({user['role']}, {user['id']})")

        if ss.get("welcome"):
            st.success(ss.pop("welcome"))

        {"Admin": admin_dashboard,
         "Teacher": teacher_dashboard,
         "Student": student_dashboard}[user["role"]](user)
        return

    # not logged in: the login "card" in the middle of the screen -
    # the middle column centres it left/right (full width on a phone),
    # vertical_alignment="center" centres it top/bottom (with the CSS height at the top of the file)
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        with st.container(key="login_box", vertical_alignment="center"):
            login_screen()


def login_screen():
    """Everything on the Login page before logging in: Log in / Sign up."""

    ss = st.session_state

    st.header("🔐 Login")

    # a teacher sign-up was cancelled (student face) -> back to "Log in", empty form
    if ss.pop("go_to_login", False):
        ss.login_action = "Log in"
        for key in ("signup_name", "signup_user", "signup_pass", "signup_pass2", "signup_classes"):
            ss.pop(key, None)
        st.error("⛔ Registration cancelled: this face belongs to a registered student. "
                 "That account has been frozen and the admin has been informed.")

    # a sign-up was saved -> back to "Log in" (it must be approved first), empty form
    if ss.get("signup_done"):
        ss.login_action = "Log in"
        for key in ("signup_name", "signup_user", "signup_pass", "signup_pass2", "signup_classes"):
            ss.pop(key, None)
        st.success(ss.pop("signup_done"))

    action = st.segmented_control("What do you want to do?", ["Log in", "Sign up"],
                                  default="Log in", required=True, key="login_action")

    if action == "Sign up":
        st.subheader("📝 Sign up – register your face")
        signup()
        return

    # a student typed the ID -> now the face check
    if ss.get("login_pending"):
        student_face_login()
        return

    role = st.segmented_control("Login as", ["Teacher", "Student"],
                                default="Student", required=True, key="login_role")

    with st.form("login_form"):

        if role == "Student":
            # "ID_" is fixed, the student types only the number
            c1, c2 = st.columns([1, 6], vertical_alignment="bottom")
            c1.markdown("### ID_")
            number = c2.text_input("Student number", placeholder="001", key="login_number")
        else:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

        submitted = st.form_submit_button("Continue" if role == "Student" else "Login",
                                          type="primary")

    if role == "Teacher":
        st.caption("No account yet? Sign up as a teacher, or ask the admin.")
    elif role == "Student":
        st.caption("Only the number – e.g. **7** or **007** for ID_007. "
                   "Next, the camera checks your face – it is your password.")

    if not submitted:
        return

    if role == "Teacher":
        # the admin has no login option of its own: the admin's username + password
        # typed into the Teacher form opens the admin dashboard
        if auth.check_admin(username, password):
            log_in({"role": "Admin", "id": "ADMIN", "name": "Admin"})
            st.rerun()

        account = store.get_teacher_account(username) if username.strip() else None
        teacher = store.get_person(account["person_id"]) if account else None

        if teacher and is_rejected(teacher):
            teacher = None                     # login rejected: answer like a wrong login

        if teacher and auth.check_password(password, account["password_hash"]):
            if is_pending(teacher):
                st.warning("⏳ Your sign-up is waiting for approval by the admin.")
                return
            log_in({"role": "Teacher", "id": teacher["id"], "name": teacher["name"]})
            st.rerun()
        st.error("Wrong username or password")

    else:
        student_id = student_id_from(number)

        if student_id is None:
            st.error("Enter the number of your Student ID, e.g. 001")
            return

        person = store.get_person(student_id)

        if person and is_rejected(person):
            person = None                      # login rejected: answer "Invalid Student ID"

        if person and person["role"] == "student" and person.get("frozen"):
            st.error("🔒 This account is frozen – please contact the admin.")
            return

        if person and person["role"] == "student" and is_pending(person):
            st.warning("⏳ Your sign-up is waiting for approval by the admin "
                       "or a teacher of your class.")
            return

        if person and person["role"] == "student":
            ss.login_pending = person
            reset_face_login()
            st.rerun()
        st.error("Invalid Student ID")


def personal_attendance(report, person_id, label):
    """Metrics, chart and records for one person (used by Student and Teacher)."""

    mine = report[report["ID"] == person_id]
    counts = count_status(this_month(mine))
    total = sum(counts.values())
    percent = round(counts["Present"] / total * 100, 1) if total else 0

    st.subheader("This month")
    c1, c2 = st.columns([3, 1])
    with c1:
        show_status_metrics(counts)
    c2.metric("Attendance %", f"{percent}%")

    status_bar_chart(counts, label)

    st.subheader("All records")

    if mine.empty:
        st.info("No attendance records yet")
    else:
        st.dataframe(mine.sort_values("Date", ascending=False), hide_index=True, width="stretch")


# ---------------- STUDENT ---------------- #

def student_dashboard(user):

    person = store.get_person(user["id"]) or {}
    if person.get("year"):
        st.caption(f"Class: **{class_name(person['year'], person['division'])}**")

    view = st.segmented_control("Show", ["📷 Mark my attendance", "📊 My attendance"],
                                default="📷 Mark my attendance", required=True, key="student_view")

    if view == "📷 Mark my attendance":
        mark_my_attendance(user)
    else:
        personal_attendance(full_report(), user["id"], "My attendance this month")


# ---------------- TEACHER (view only) ---------------- #

def teacher_dashboard(user):

    waiting = class_students(user["id"], state="pending")
    rejected = class_students(user["id"], state="rejected")

    view = st.segmented_control("Show", ["📷 Mark my attendance", "👨‍🏫 My attendance",
                                         "🎓 My students", "🚩 Flagged attendance",
                                         "🖊 Approvals"],
                                default="📷 Mark my attendance", required=True, key="teacher_view")

    if waiting and view != "🖊 Approvals":
        st.info(f"🖊 {len(waiting)} student sign-up(s) of your classes are waiting for approval.")

    if view == "📷 Mark my attendance":
        mark_my_attendance(user)
    elif view == "👨‍🏫 My attendance":
        personal_attendance(full_report(), user["id"], "My attendance this month")
    elif view == "🚩 Flagged attendance":
        flagged_attendance(user)
    elif view == "🖊 Approvals":
        st.caption("Students of your classes. Approve = they can log in. "
                   "Reject login = they get \"Invalid Student ID\" (data kept, can be re-approved). "
                   "Delete all data = removed completely.")
        approval_list(waiting, rejected, "teacher")
    else:
        my_students(user)


def person_card_details(p, accounts):
    """Class (students) or username + classes (teachers) under a name."""

    if p["role"] == "student" and p.get("year"):
        st.caption(f"Class: {class_name(p['year'], p['division'])}")
    elif p["role"] == "teacher":
        classes = ", ".join(class_name(c["year"], c["division"])
                            for c in store.get_teacher_classes(p["id"])) or "none"
        st.caption(f"Username: {accounts.get(p['id'], '–')} · Classes: {classes}")


def delete_all_button(p, column, who):
    """Delete the person with ALL their data - only after ticking the box."""

    sure = column.checkbox("Yes, delete everything", key=f"{who}_delete_ok_{p['id']}")
    if column.button("🗑 Delete all data", key=f"{who}_delete_{p['id']}", disabled=not sure):
        store.delete_person(p["id"])            # face data, attendance, account, logins
        st.rerun()


def approval_list(pending, rejected, who):
    """
    Waiting sign-ups: ✅ Approve / ⛔ Reject login / 🗑 Delete all data.
    Rejected logins:  ♻ Re-approve login / 🗑 Delete all data.
    who = 'admin' or 'teacher' (for the widget keys).
    """

    accounts = {a["person_id"]: a["username"] for a in store.get_teacher_accounts()}

    if not pending:
        st.success("No sign-ups waiting for approval.")

    for p in pending:
        with st.container(border=True):
            st.markdown(f"**⏳ {p['name']}** ({p['id']}, {p['role']}) – signed up {p['registered_on']}")
            person_card_details(p, accounts)

            c1, c2, c3 = st.columns(3, vertical_alignment="top")

            if c1.button("✅ Approve", key=f"{who}_approve_{p['id']}", type="primary"):
                store.set_approved(p["id"], True)
                st.rerun()

            if c2.button("⛔ Reject login", key=f"{who}_reject_{p['id']}",
                         help="Keeps all data. The login answers 'Invalid ID'. "
                              "Can be re-approved later."):
                store.reject_login(p["id"])
                store.delete_login_sessions_of(p["id"])
                st.rerun()

            delete_all_button(p, c3, who)

    # ---- rejected logins: can be re-approved ----
    if rejected:
        st.subheader("⛔ Login rejected")
        st.caption("All their data is still there. Their face stays registered, so they can't "
                   "sign up again with a new ID – use Delete all data for that.")

    for p in rejected:
        with st.container(border=True):
            st.markdown(f"**⛔ {p['name']}** ({p['id']}, {p['role']}) – signed up {p['registered_on']}")
            person_card_details(p, accounts)

            c1, c2 = st.columns(2, vertical_alignment="top")

            if c1.button("♻ Re-approve login", key=f"{who}_reapprove_{p['id']}", type="primary"):
                store.set_approved(p["id"], True)
                st.rerun()

            delete_all_button(p, c2, who)


def record_points(record):
    """The saved GPS readings of a record as map points (red)."""

    try:
        saved = json.loads(record.get("gps_readings") or "{}")
    except ValueError:
        return []

    readings = clean_readings(saved.get("readings")) if isinstance(saved, dict) else []
    return [{"lat": r["lat"], "lon": r["lon"], "color": "#d03b3b"} for r in readings]


def flagged_attendance(user):
    """Flagged records of the teacher's class students: reason, map, Approve / Reject."""

    students = {s["id"]: s for s in class_students(user["id"])}
    rows = [r for r in store.get_attendance("student")
            if r["person_id"] in students and r.get("flagged") and not r.get("review")]

    if not students:
        st.info("You have no classes yet. Ask the admin to assign them (Admin → People).")
        return
    if not rows:
        st.success("No flagged attendance. 🎉")
        return

    st.caption("Approve = counts as normal attendance. Reject = counts as Absent "
               "(the record stays as evidence).")
    cfg = college_settings()

    for r in sorted(rows, key=lambda r: (str(r["date"]), r["person_id"]), reverse=True):
        s = students[r["person_id"]]
        key = f"{r['person_id']}_{r['date']}"

        with st.container(border=True):
            st.markdown(f"**🚩 {s['name']}** ({s['id']}, "
                        f"{class_name(s['year'], s['division'])}) – {r['date']}")
            st.caption(f"Time In {r.get('time_in') or '–'} · Time Out {r.get('time_out') or '–'}"
                       f" · IP {r.get('ip_address') or '–'}")
            st.warning(f"Reason: {r.get('flag_reason') or 'unknown'}")

            # red = the student's 3 GPS readings, blue = the college centre
            points = record_points(r)
            if cfg["lat"] is not None:
                points.append({"lat": cfg["lat"], "lon": cfg["lon"], "color": "#1f77b4"})
            if points:
                st.map(pd.DataFrame(points), latitude="lat", longitude="lon",
                       color="color", size=8, zoom=15)

            c1, c2 = st.columns(2)
            if c1.button("✅ Approve", key=f"approve_{key}", type="primary", width="stretch"):
                store.set_review(r["person_id"], str(r["date"]), "approved")
                st.rerun()
            if c2.button("❌ Reject", key=f"reject_{key}", width="stretch"):
                store.set_review(r["person_id"], str(r["date"]), "rejected")
                st.rerun()


def my_students(user):
    """Students of the teacher's classes, filtered by year and branch."""

    classes = store.get_teacher_classes(user["id"])

    if not classes:
        st.info("You have no classes yet. Ask the admin to assign them (Admin → People).")
        return

    st.caption("Your classes: " + ", ".join(
        class_name(c["year"], c["division"])
        for c in sorted(classes, key=lambda c: (YEARS.index(c["year"]), c["division"]))))

    # ---- selection first ----
    c1, c2 = st.columns(2)
    year = c1.selectbox("Year", ["All"] + sorted({c["year"] for c in classes}, key=YEARS.index),
                        key="my_year")
    division = c2.selectbox("Branch", ["All"] + sorted({c["division"] for c in classes}),
                            format_func=branch_label, key="my_div")

    students = [s for s in class_students(user["id"])
                if year in ("All", s["year"]) and division in ("All", s["division"])]
    class_of = {s["id"]: class_name(s["year"], s["division"]) for s in students}

    report = full_report()
    rows = report[report["ID"].isin(list(class_of))]
    rows = rows.assign(Class=rows["ID"].map(class_of))
    month = this_month(rows)

    # ---- then metrics, charts and the table ----
    st.subheader(f"Students this month ({len(students)} students)")
    show_status_metrics(count_status(month))

    col1, col2 = st.columns(2)
    with col1:
        status_bar_chart(count_status(month), "My students, this month")
    with col2:
        st.markdown("**Days per student, this month**")
        per_person_chart(month)

    st.subheader("All records")

    if rows.empty:
        st.info("No attendance records yet")
    else:
        st.dataframe(rows.sort_values(["Date", "Name"], ascending=[False, True]),
                     hide_index=True, width="stretch")

    st.download_button("⬇ Download as Excel", data=excel_bytes(rows),
                       file_name=f"my_students_{rules.now_local().date().isoformat()}.xlsx",
                       disabled=rows.empty)


# ---------------- ADMIN (create / read / update / delete) ---------------- #

def admin_dashboard(user):

    # if the online database can't be reached, still show the Database tab
    # so the admin can switch back to offline
    try:
        report = full_report()
        people = store.get_people()
        alerts = store.get_alerts()
        error = None
    except Exception as e:
        alerts, error = [], e

    open_alerts = [a for a in alerts if a["status"] == "open"]
    pending = [p for p in people if is_pending(p)] if not error else []
    rejected = [p for p in people if is_rejected(p)] if not error else []

    if open_alerts:
        st.error(f"🚨 **{len(open_alerts)} frozen account(s) need a decision** – "
                 "see the 🚨 Alerts tab.")
    if pending:
        st.warning(f"🖊 **{len(pending)} sign-up(s) waiting for approval** – "
                   "see the 🖊 Approvals tab.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["📊 Overview", "👥 People", "🗓 Attendance records", "🔑 Teacher accounts", "🗄 Database",
         "⚙️ App settings", f"🚨 Alerts ({len(open_alerts)})",
         f"🖊 Approvals ({len(pending)})" + (f" · ⛔ {len(rejected)}" if rejected else ""),
         "🔁 Retrain face"])

    with tab5:
        admin_database()

    with tab6:
        admin_settings()

    if error:
        for tab in (tab1, tab2, tab3, tab4, tab7, tab8, tab9):
            with tab:
                db_error(error)
        return

    with tab9:
        admin_retrain(people)

    with tab7:
        admin_alerts(alerts, people)

    with tab8:
        # st.caption("New sign-ups can log in only after approval. Reject login = the login says "
        #            "\"Invalid ID\" (data kept, can be re-approved). Delete all data = removed completely.")
        approval_list(pending, rejected, "admin")

    with tab1:
        month = this_month(report)
        show_status_metrics(count_status(month))

        col1, col2 = st.columns(2)
        with col1:
            status_bar_chart(count_status(month), "Everyone, this month")
        with col2:
            st.markdown("**Days per person, this month**")
            per_person_chart(month)

    with tab2:
        admin_people(people)

    with tab3:
        admin_records(report, people)

    with tab4:
        admin_teacher_accounts(people)


def admin_alerts(alerts, people):
    """Students frozen because their face was used in a teacher sign-up: Restore or Delete."""

    by_id = {p["id"]: p for p in people}
    open_alerts = [a for a in alerts if a["status"] == "open"]

    def close_alerts(student_id, status):
        # one decision closes every open alert of that student
        for a in open_alerts:
            if a["student_id"] == student_id:
                store.set_alert_status(a["id"], status)

    if not open_alerts:
        st.success("No open alerts.")

    for a in open_alerts:
        student = by_id.get(a["student_id"])

        with st.container(border=True):
            st.markdown(f"**🔒 {person_label(student) if student else a['student_id']}**")
            st.caption(f"{a['created_at']} · {a['message']}")

            if student is None:
                st.caption("This student no longer exists.")
                if st.button("Close alert", key=f"alert_close_{a['id']}"):
                    store.set_alert_status(a["id"], "deleted")
                    st.rerun()
                continue

            c1, c2, c3 = st.columns(3, vertical_alignment="center")

            if c1.button("♻ Restore", key=f"alert_restore_{a['id']}", type="primary"):
                store.set_frozen(student["id"], False)
                close_alerts(student["id"], "restored")
                st.rerun()

            sure = c2.checkbox("Yes, delete this student", key=f"alert_del_ok_{a['id']}")

            if c3.button("🗑 Delete student", key=f"alert_del_{a['id']}", disabled=not sure):
                store.delete_person(student["id"])      # face, attendance, logins
                close_alerts(student["id"], "deleted")
                st.rerun()

    handled = [a for a in alerts if a["status"] != "open"]

    if handled:
        st.subheader("History")
        st.dataframe([{"Time": a["created_at"], "Student": a["student_id"],
                       "What happened": a["message"], "Decision": a["status"]} for a in handled],
                     hide_index=True, width="stretch")


# ---------------- ADMIN: RETRAIN FACE ---------------- #

def reset_retrain(person_id=None):
    """Forget the new photos (and remember which person they are for)."""

    ss = st.session_state
    ss.rt_person = person_id
    ss.rt_embs = []
    ss.rt_last = 0.0
    ss.rt_active = False                       # the camera starts only after "Start camera"
    ss.rt_cam_key = ss.get("rt_cam_key", 0) + 1
    ss.rt_cam = ("Look at the camera", "searching")


def delete_face_data(person_id):
    """Remove only this person's face photos (Retrain -> Replace). Works offline and online."""

    if hasattr(store, "client"):               # online: Supabase
        store.client.table("embeddings").delete().eq("person_id", person_id).execute()
    else:                                      # offline: the SQLite file
        with store._connect() as db:
            db.execute("DELETE FROM embeddings WHERE person_id = ?", (person_id,))


def add_retrain_photo(person, emb):
    """
    One new photo for `person`. Returns (ok, message).
    The face must NOT belong to another registered person, and all new photos
    must show the same person (same rules as sign-up).
    """

    ss = st.session_state
    ids, embs = store.get_all_embeddings()
    others = ids != person["id"]               # compare only with everybody else

    ok, message, existing_id = auto_capture.check_registration_photo(
        emb, ss.rt_embs, ids[others], embs[others])

    if existing_id:
        other = store.get_person(existing_id) or {"name": existing_id, "id": existing_id}
        message = (f"⚠ This face belongs to {other['name']} ({other['id']}) – "
                   "retraining stopped, no photo was kept.")
        reset_retrain(person["id"])

    if ok:
        ss.rt_embs.append(emb)
        ss.rt_last = time.time()

    return ok, message


@st.fragment
def retrain_auto(person):
    """Live camera: takes the new photos by itself (same checks as sign-up)."""

    ss = st.session_state
    if ss.get("rt_person") != person["id"] or not ss.get("rt_active"):
        ss.pop("retrain_live__frame", None)    # late frame from another person / stopped camera
        return

    frame = live_camera.take_frame("retrain_live")

    if frame:
        seq, img = frame
        ss.rt_ack = seq

        face, message = face_engine.get_face(get_model(), img)
        status = "searching"

        if face is not None:
            message = face_engine.registration_checks(face, img)
            status = "error" if message else "ok"

        if face is not None and message is None:
            emb = face.normed_embedding
            message = auto_capture.auto_registration_ready(emb, ss.rt_embs, ss.rt_last, time.time())

            if message is None:
                ok, message = add_retrain_photo(person, emb)

                if not ok:
                    ss.rt_error = message
                    st.rerun(scope="app")       # show the error on the full page

                if len(ss.rt_embs) == REGISTER_PHOTOS:
                    st.rerun(scope="app")       # all photos taken -> show the Save buttons

                message = f"📸 {len(ss.rt_embs)} / {REGISTER_PHOTOS} – now turn your head slightly"

        ss.rt_cam = (message, status)

    taken = len(ss.rt_embs)
    st.progress(taken / REGISTER_PHOTOS, text=f"New photos: {taken} / {REGISTER_PHOTOS}")

    message, status = ss.rt_cam
    live_camera.live_camera("retrain_live", active=True, message=message, status=status,
                            ack=ss.get("rt_ack", 0))


def retrain_manual(person):
    """Click 'Take photo' for each new photo."""

    ss = st.session_state
    taken = len(ss.rt_embs)
    st.progress(taken / REGISTER_PHOTOS, text=f"New photos: {taken} / {REGISTER_PHOTOS}")

    photo = st.camera_input(f"Photo {taken + 1}", key=f"rt_cam_{ss.rt_cam_key}")
    if photo is None:
        return

    with st.spinner("Checking face..."):
        img = face_engine.decode_image(photo.getvalue())
        emb, message = face_engine.get_face_embedding(get_model(), img)

    if emb is None:
        st.error(f"{message} Click **Clear photo** and try again.")
        return

    ok, message = add_retrain_photo(person, emb)
    if not ok:
        st.error(message)
        return

    ss.rt_cam_key += 1                         # a fresh camera for the next photo
    st.rerun()


def admin_retrain(people):
    """Take new face photos for someone whose face is not recognised any more."""

    ss = st.session_state

    # st.caption("Use this when someone's face is not recognised any more (new glasses, beard, "
            #    "haircut, poor first photos). Their ID, attendance and login stay the same.")

    if ss.get("rt_done"):
        st.success(ss.pop("rt_done"))
    if ss.get("rt_error"):
        st.error(ss.pop("rt_error"))

    if not people:
        st.info("Nobody is registered yet")
        return

    person = st.selectbox("Whose face?", people, format_func=person_label, key="rt_choose")

    if ss.get("rt_person") != person["id"]:
        reset_retrain(person["id"])            # another person chosen -> start fresh

    ids, _ = store.get_all_embeddings()
    st.caption(f"{person['name']} has **{int((ids == person['id']).sum())}** face photo(s) now.")

    # ---- taking the new photos ----
    if len(ss.rt_embs) < REGISTER_PHOTOS:

        if not ss.rt_active:
            st.caption("The person stands in front of this device. "
                       "The camera starts only after the button.")
            if st.button("📷 Start camera", key="rt_start", type="primary"):
                ss.rt_active = True
                st.rerun()
            return

        st.info(f"We take {REGISTER_PHOTOS} photos of **{person['name']}**. "
                "Between photos, **turn the head slightly** or move a little.")

        if capture_mode("rt_mode") == "Auto":
            retrain_auto(person)
        else:
            retrain_manual(person)

        if st.button("✖ Stop camera", key="rt_stop"):
            reset_retrain(person["id"])
            st.rerun()
        return

    # ---- all photos taken: add or replace ----
    st.success(f"All {REGISTER_PHOTOS} new photos taken for **{person['name']}**.")

    def finish():
        ids, _ = store.get_all_embeddings()
        ss.rt_done = (f"✅ Face retrained for {person['name']}: "
                      f"{int((ids == person['id']).sum())} face photo(s) now.")
        reset_retrain(person["id"])
        st.rerun()

    c1, c2, c3 = st.columns(3, vertical_alignment="top")

    if c1.button("➕ Add to the old photos", key="rt_add", type="primary",
                 help="Safest: the old photos stay, recognition gets more examples."):
        store.save_embeddings(person["id"], ss.rt_embs)
        finish()

    sure = c2.checkbox("Yes, replace", key=f"rt_replace_ok_{ss.rt_cam_key}")   # new tick each round
    if c2.button("♻ Replace the old photos", key="rt_replace", disabled=not sure,
                 help="The old face photos are deleted, only the new ones are kept."):
        delete_face_data(person["id"])
        store.save_embeddings(person["id"], ss.rt_embs)
        finish()

    if c3.button("Start over", key="rt_restart"):
        reset_retrain(person["id"])
        st.rerun()


def admin_people(people):

    st.caption("People are added by signing up on the **Login** page (a face is needed) "
               "and approved in the 🖊 Approvals tab.")

    if not people:
        st.info("Nobody is registered yet")
        return

    person = st.selectbox("Choose a person", people, format_func=person_label, key="adm_person")

    # ---- UPDATE ----
    with st.form("edit_person"):
        st.markdown("**Edit**")
        name = st.text_input("Name", value=person["name"])
        role = st.radio("Role", ["student", "teacher"], horizontal=True,
                        index=0 if person["role"] == "student" else 1)

        # students: year + branch ("–" = not set). An old value (e.g. division "A")
        # stays in the list as "A (old)" so it can be changed to a branch.
        c1, c2 = st.columns(2)
        year_options = ["–"] + YEARS
        div_options = ["–"] + storage.get_branches()
        if person.get("division") and person["division"] not in div_options:
            div_options.append(person["division"])
        year = c1.selectbox("Year (students)", year_options,
                            index=year_options.index(person.get("year") or "–"))
        division = c2.selectbox("Branch (students)", div_options, format_func=branch_label,
                                index=div_options.index(person.get("division") or "–"))

        # teachers: classes they teach (old classes stay selectable)
        current = [(c["year"], c["division"]) for c in store.get_teacher_classes(person["id"])]
        classes = st.multiselect("Classes taught (teachers)", all_classes(extra=current),
                                 default=current, format_func=class_label)

        if st.form_submit_button("💾 Save changes"):
            if not name.strip():
                st.error("Name can't be empty")
            elif role == "student" and "–" in (year, division):
                st.error("Choose the student's Year and Branch")
            else:
                store.update_person(person["id"], name.strip(), role)
                if role == "student":
                    store.set_class(person["id"], year, division)
                else:
                    store.set_teacher_classes(person["id"], classes)
                st.success("Saved")
                st.rerun()

    # ---- RE-APPROVE a rejected login ----
    if is_rejected(person):
        with st.container(border=True):
            st.markdown("**⛔ Login rejected** – the login answers \"Invalid ID\".")
            if st.button("♻ Re-approve login", key=f"people_reapprove_{person['id']}"):
                store.set_approved(person["id"], True)
                st.rerun()

    # ---- DELETE ----
    with st.container(border=True):
        st.markdown("**Delete**")
        sure = st.checkbox(f"Yes, delete {person['name']} with their face data, "
                           "attendance and teacher login", key=f"del_ok_{person['id']}")

        if st.button("🗑 Delete person", disabled=not sure, type="primary"):
            store.delete_person(person["id"])
            st.rerun()

    st.subheader("All people")
    st.dataframe(people, hide_index=True, width="stretch")


def admin_records(report, people):

    # ---- READ: choose dates (and a person) first, the table comes below ----
    st.subheader("🔎 View records")

    today = rules.now_local().date()
    col1, col2, col3, col4 = st.columns(4)
    start = col1.date_input("From", value=today.replace(day=1), key="view_from")
    end = col2.date_input("To", value=today, key="view_to")
    group = col3.selectbox("Students / Teachers", ["Everyone", "Students", "Teachers"],
                           key="view_role")
    who = col4.selectbox("Person", ["Everyone"] + people, key="view_person",
                         format_func=lambda p: p if p == "Everyone" else person_label(p))

    if start > end:
        st.warning("'From' date must be before 'To' date")
    else:
        shown = filter_dates(report, start, end)
        if group != "Everyone":
            shown = shown[shown["Role"] == ("student" if group == "Students" else "teacher")]
        if who != "Everyone":
            shown = shown[shown["ID"] == who["id"]]

        show_status_metrics(count_status(shown))

        if shown.empty:
            st.info("No attendance records for these dates")
        else:
            st.dataframe(shown.sort_values(["Date", "Name"], ascending=[False, True]),
                         hide_index=True, width="stretch")

        st.download_button("⬇ Download these records as Excel", data=excel_bytes(shown),
                           file_name=f"attendance_{start.isoformat()}_to_{end.isoformat()}.xlsx",
                           disabled=shown.empty)

    if not people:
        return

    st.divider()
    st.subheader("✏️ Add / edit / delete a record")

    col1, col2 = st.columns(2)
    person = col1.selectbox("Person", people, format_func=person_label, key="rec_person")
    day = col2.date_input("Date", value=rules.now_local().date(), key="rec_date").isoformat()

    record = store.get_record(person["id"], day) or {}

    with st.form("edit_record"):
        c1, c2 = st.columns(2)
        time_in = c1.text_input("Time In (HH:MM:SS)", value=record.get("time_in") or "")
        time_out = c2.text_input("Time Out (HH:MM:SS)", value=record.get("time_out") or "")

        if st.form_submit_button("💾 Save record"):
            time_in, time_out = time_in.strip(), time_out.strip()

            if not (valid_time(time_in) and valid_time(time_out)):
                st.error("Use the format HH:MM:SS, for example 09:05:00")
            elif time_out and not time_in:
                st.error("Time Out needs a Time In")
            elif time_out and time_out < time_in:
                st.error("Time Out can't be before Time In")
            elif not time_in:
                st.error("Enter at least a Time In (or delete the record)")
            else:
                store.save_record(person["id"], day, time_in, time_out)
                st.success("Saved")
                st.rerun()

    if record:
        if st.button("🗑 Delete this record"):
            store.delete_record(person["id"], day)
            st.rerun()
    else:
        st.caption("No record for this person on this day yet – saving will add one.")


def admin_teacher_accounts(people):

    accounts = store.get_teacher_accounts()
    names = {p["id"]: p["name"] for p in people}

    # one account per teacher: only teachers WITHOUT an account can get one
    has_account = {a["person_id"] for a in accounts}
    teachers = [p for p in people if p["role"] == "teacher" and p["id"] not in has_account]

    # ---- CREATE ----
    with st.form("add_account", clear_on_submit=True):
        st.markdown("**Add an account**")

        if not teachers and has_account:
            st.caption("Every teacher already has an account – use Reset password below.")
        elif not teachers:
            st.caption("A teacher must sign up on the Login page first.")

        teacher = st.selectbox("Teacher", teachers, format_func=person_label, key="acc_teacher")
        username = st.text_input("Username")
        password = st.text_input("Password (min. 6 characters)", type="password")

        if st.form_submit_button("➕ Create account", disabled=not teachers):
            existing = store.get_account_of_teacher(teacher["id"])

            if not username.strip() or len(password) < 6:
                st.error("Enter a username and a password of at least 6 characters")
            elif existing:
                st.error(f"This teacher already has an account ({existing['username']})")
            elif store.get_teacher_account(username) or auth.is_admin_username(username):
                st.error("This username is already taken")
            else:
                store.save_teacher_account(username, auth.hash_password(password), teacher["id"])
                st.success("Account created")
                st.rerun()

    if accounts:
        account = st.selectbox("Choose an account", accounts,
                               format_func=lambda a: f"{a['username']} – {names.get(a['person_id'], '?')}")

        # ---- UPDATE (reset password) ----
        with st.form("reset_password", clear_on_submit=True):
            st.markdown("**Reset password**")
            new_password = st.text_input("New password (min. 6 characters)", type="password")

            if st.form_submit_button("🔁 Reset password"):
                if len(new_password) < 6:
                    st.error("Password must have at least 6 characters")
                else:
                    store.save_teacher_account(account["username"], auth.hash_password(new_password),
                                               account["person_id"])
                    st.success("Password changed")

        # ---- DELETE ----
        if st.button(f"🗑 Delete account '{account['username']}'"):
            store.delete_teacher_account(account["username"])
            st.rerun()

    # ---- READ: the table comes below the forms ----
    st.subheader("All teacher accounts")

    if accounts:
        st.dataframe([{"Username": a["username"], "Teacher": names.get(a["person_id"], "?"),
                       "ID": a["person_id"]} for a in accounts],
                     hide_index=True, width="stretch")
    else:
        st.info("No teacher accounts yet")


def admin_settings():
    """Settings for the whole app, saved in settings.json."""

    st.subheader("⚙️ App settings")

    with st.form("app_settings"):
        capture = st.segmented_control(
            "Default capture mode", ["Auto", "Manual"],
            default=storage.get_capture_default(), required=True, key="set_capture",
            help="Where the Capture mode switch starts on every camera. "
                 "People can still change it themselves.")

        show_switch = st.toggle(
            "Show the Capture mode switch to everyone",
            value=storage.get_capture_switch_visible(), key="set_capture_switch",
            help="Off = nobody can change it; every camera uses the default above.")

        st.markdown("**Cooldown** – how long to wait after Time In before Time Out can be marked")
        c1, c2 = st.columns(2)
        student_min = c1.number_input("Students (minutes)", min_value=0.5, max_value=720.0, step=0.5,
                                      value=storage.get_cooldown("student") / 60, key="set_cd_student")
        teacher_min = c2.number_input("Teachers (minutes)", min_value=0.5, max_value=720.0, step=0.5,
                                      value=storage.get_cooldown("teacher") / 60, key="set_cd_teacher")

        if st.form_submit_button("💾 Save settings", type="primary"):
            storage.save_setting("capture_mode", capture)
            storage.save_setting("capture_switch", show_switch)
            storage.save_setting("cooldown_student", round(student_min * 60))
            storage.save_setting("cooldown_teacher", round(teacher_min * 60))
            st.success("Settings saved. They apply to the next scan / camera.")

    st.divider()
    admin_branches()

    st.divider()
    admin_college_settings()


def admin_branches():
    """The branch list: students pick one at sign-up, teachers teach Year x Branch."""

    ss = st.session_state
    branches = storage.get_branches()

    st.subheader("🏫 Branches")
    st.caption("Students choose their branch at sign-up; teachers teach Year-Branch classes "
               "(e.g. SE-Computer).")

    if ss.get("branch_msg"):
        st.info(ss.pop("branch_msg"))
    if not branches:
        st.warning("No branches yet – students can't sign up until you add one.")

    # how many people use each branch (students + teachers' classes)
    used = {}
    for p in store.get_people():
        if p["role"] == "student" and p.get("division"):
            used[p["division"]] = used.get(p["division"], 0) + 1
    for c in store.get_teacher_classes():
        used[c["division"]] = used.get(c["division"], 0) + 1

    for b in branches:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f"**{b}**" + (f" – used {used[b]}×" if used.get(b) else ""))

        if c2.button("🗑 Remove", key=f"branch_remove_{b}"):
            storage.save_branches([x for x in branches if x != b])
            ss.branch_msg = f"'{b}' removed."
            if used.get(b):
                ss.branch_msg += (f" It is still used {used[b]}× – those people keep it until you "
                                  "change them in 👥 People.")
            st.rerun()

    with st.form("add_branch", clear_on_submit=True):
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        new = c1.text_input("New branch", max_chars=40, key="branch_new")

        if c2.form_submit_button("➕ Add branch"):
            name = " ".join(new.split())                  # no extra spaces

            if not name:
                st.error("Enter a branch name.")
            elif name.lower() in {b.lower() for b in branches}:
                st.error(f"'{name}' is already in the list.")
            else:
                storage.save_branches(branches + [name])
                ss.branch_msg = f"'{name}' added."
                st.rerun()


def admin_college_settings():
    """Geofence (centre + radius) and college Wi-Fi (ALLOWED_IPS, WIFI_REQUIRED)."""

    ss = st.session_state
    cfg = college_settings()
    v = ss.get("college_form_v", 0)          # new number -> the form shows freshly saved values

    st.subheader("📍 College location & Wi-Fi")

    if ss.get("college_msg"):
        st.success(ss.pop("college_msg"))
    if cfg["gps_required"] and cfg["lat"] is None:
        st.warning("The college location is not set – students can't mark attendance until "
                   "you set it (use the 📍 button below while you are at college).")

    with st.form("college_settings"):
        gps = st.toggle("GPS location check (geofence + fake-GPS checks)",
                        value=cfg["gps_required"], key=f"college_gps_{v}",
                        help="Off = no GPS at login / Mark my attendance. "
                             "The face (and the Wi-Fi rule below) are still checked.")

        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("College latitude", value=cfg["lat"], format="%.6f",
                              min_value=-90.0, max_value=90.0, key=f"college_lat_{v}")
        lon = c2.number_input("College longitude", value=cfg["lon"], format="%.6f",
                              min_value=-180.0, max_value=180.0, key=f"college_lon_{v}")
        radius = c3.number_input("Radius (metres)", value=cfg["radius_m"], min_value=20.0,
                                 max_value=5000.0, step=10.0, key=f"college_radius_{v}")

        ips_text = st.text_area("ALLOWED_IPS – college Wi-Fi addresses (one per line, "
                                "a single IP or a range like 203.0.113.0/24)",
                                value="\n".join(cfg["allowed_ips"]), key=f"college_ips_{v}")
        wifi = st.toggle("WIFI_REQUIRED – block attendance when not on the college Wi-Fi",
                         value=cfg["wifi_required"], key=f"college_wifi_{v}")

        if st.form_submit_button("💾 Save college settings", type="primary"):
            ips = [line.strip() for line in ips_text.splitlines() if line.strip()]
            bad = [ip for ip in ips if not checks.ip_allowed(ip.split("/")[0], [ip])]

            if gps and (lat is None or lon is None):
                st.error("Enter the college latitude and longitude (needed for the GPS check).")
            elif bad:
                st.error("Not a valid IP / range: " + ", ".join(bad))
            elif wifi and not ips:
                st.error("Add at least one college IP before turning WIFI_REQUIRED on.")
            else:
                if lat is not None and lon is not None:
                    storage.save_setting("college_lat", lat)
                    storage.save_setting("college_lon", lon)
                storage.save_setting("geofence_radius_m", radius)
                storage.save_setting("gps_required", gps)
                storage.save_setting("allowed_ips", ips)
                storage.save_setting("wifi_required", wifi)
                ss.college_form_v = v + 1
                ss.college_msg = "College settings saved."
                st.rerun()

    # ---- buttons that read the admin's CURRENT IP / location (be at college!) ----
    my_ip = current_ip()
    st.caption(f"Your current IP address: **{my_ip}**")

    c1, c2 = st.columns(2)

    if c1.button("🌐 Use my current IP", width="stretch",
                 help="Connect to the college Wi-Fi first, then press this."):
        if my_ip not in cfg["allowed_ips"]:
            storage.save_setting("allowed_ips", cfg["allowed_ips"] + [my_ip])
        ss.college_form_v = v + 1
        ss.college_msg = f"{my_ip} added to ALLOWED_IPS."
        st.rerun()

    if c2.button("📍 Use my current location", width="stretch",
                 help="Stand at the college, then press this. It becomes the geofence centre."):
        ss.admin_loc_on = True
        reset_location("admin_loc")
        st.rerun()

    if ss.get("admin_loc_on"):
        readings = location_step("admin_loc")
        if readings:
            lat, lon = checks.average_position(readings)
            storage.save_setting("college_lat", round(lat, 6))
            storage.save_setting("college_lon", round(lon, 6))
            ss.admin_loc_on = False
            reset_location("admin_loc")
            ss.college_form_v = v + 1
            ss.college_msg = f"College centre set to {lat:.6f}, {lon:.6f}."
            st.rerun()


DB_LABELS = {"offline": "💻 Offline (SQLite file)", "online": "☁️ Online (Supabase)"}


def db_error(e):
    st.error(f"Could not read the database: {e}\n\n"
             "Check the internet connection, the Supabase keys in secrets, "
             "and that the SQL from README.md was run in Supabase.")


def view_rows(db, table):
    """One table of a database as a DataFrame (never shows password hashes)."""

    if table == "People":
        return pd.DataFrame(db.get_people())

    if table in ("Student attendance", "Teacher attendance"):
        role = "student" if table == "Student attendance" else "teacher"
        rows = sorted(db.get_attendance(role), key=lambda r: (str(r["date"]), r["person_id"]),
                      reverse=True)
        return pd.DataFrame(rows)

    if table == "Teacher classes":
        return pd.DataFrame(db.get_teacher_classes())

    if table == "Branches":
        return pd.DataFrame({"branch": db.get_branch_list()})

    if table == "App settings":
        return pd.DataFrame([{"setting": k, "value": str(v)}
                             for k, v in sorted(db.get_settings().items())])

    if table == "Teacher accounts":
        return pd.DataFrame([{"username": a["username"], "person_id": a["person_id"]}
                             for a in db.get_teacher_accounts()])

    # face photos: only how many embeddings each person has
    ids, _ = db.get_all_embeddings()
    return (pd.Series(ids, dtype=str).value_counts().sort_index()
            .rename_axis("person_id").reset_index(name="face photos"))


def admin_database():

    online_ok = storage.online_available()

    # ---- 1. which database the whole app uses ----
    st.subheader("Storage in use")

    mode = storage.get_mode()
    chosen = st.segmented_control("The whole app saves to", list(DB_LABELS),
                                  format_func=DB_LABELS.get, default=mode,
                                  required=True, key="db_mode", disabled=not online_ok)

    if not online_ok:
        st.caption("Online is off: add SUPABASE_URL and SUPABASE_KEY to the secrets first.")
    elif chosen != mode:
        storage.set_mode(chosen)
        st.rerun()

    # ---- 2. sync ----
    st.subheader("Sync")
    st.caption("Copies what is missing to the other side. Nothing is ever deleted.")

    if online_ok:
        c1, c2 = st.columns(2)
        upload = c1.button("⬆ Upload offline → online", width="stretch")
        download = c2.button("⬇ Download online → offline", width="stretch")

        if upload or download:
            source, target = ("offline", "online") if upload else ("online", "offline")
            try:
                with st.spinner("Syncing..."):
                    r = storage.sync_stores(get_store(source), get_store(target))
            except Exception as e:
                db_error(e)
            else:
                st.success(f"Copied {r['people']} people, {r['embeddings']} face photos, "
                           f"{r['records']} attendance records, {r['classes']} teacher classes, "
                           f"{r['accounts']} teacher accounts, {r['branches']} branches, "
                           f"{r['settings']} App settings")
                for conflict in r["conflicts"]:
                    st.warning(f"Not copied: {conflict}")
    else:
        st.info("Sync needs the online database (Supabase secrets).")

    # ---- 3. view either database (selection first, table below) ----
    st.subheader("View data")

    c1, c2 = st.columns(2)
    where = c1.segmented_control("Database", list(DB_LABELS), format_func=DB_LABELS.get,
                                 default="offline", required=True, key="db_view")
    table = c2.selectbox("Table", ["People", "Student attendance", "Teacher attendance",
                                   "Teacher classes", "Teacher accounts",
                                   "Face photos per person", "Branches", "App settings"],
                         key="db_table")

    if where == "online" and not online_ok:
        st.info("The online database is not set up yet.")
        return

    try:
        rows = view_rows(get_store(where), table)
    except Exception as e:
        db_error(e)
        return

    if rows.empty:
        st.info("This table is empty")
    else:
        st.caption(f"{len(rows)} rows")
        st.dataframe(rows, hide_index=True, width="stretch")


# ======================================================================
# SIDEBAR + THE PAGE
# ======================================================================

# ---- stay logged in after a page reload ----
# The browser sends the cookie when the page loads; st.context.cookies keeps that value.
ss = st.session_state
cookie_token = st.context.cookies.get(LOGIN_COOKIE)
restore_failed = False

if not ss.get("user") and cookie_token and ss.get("checked_token") != cookie_token:
    try:
        restored = auth.user_from_token(store, cookie_token)
        ss.checked_token = cookie_token          # check each cookie only once
    except Exception:
        restored, restore_failed = None, True    # database not reachable: try again later

    if restored:
        ss.user, ss.login_token = restored, cookie_token

# a student whose account was frozen (or deleted) meanwhile is logged out at once
if ss.get("user") and ss.user["role"] == "Student" and not restore_failed:
    me = store.get_person(ss.user["id"])
    if not me or me.get("frozen"):
        log_out()

st.sidebar.title("AIMS - AI based Attendance Management System")
st.sidebar.caption(f"Database: {store.name}")
st.sidebar.caption(f"Match threshold: {MATCH_THRESHOLD}")

# write (or remove) the login cookie in the browser
with st.sidebar:
    if ss.get("user") and ss.get("login_token"):
        browser_cookie.set_cookie(LOGIN_COOKIE, ss.login_token, auth.LOGIN_DAYS * 24 * 3600)
        ss.cookie_written = True
    elif (cookie_token or ss.get("cookie_written")) and not restore_failed:
        browser_cookie.set_cookie(LOGIN_COOKIE, "", 0)      # logged out -> forget it

# Logout at the very end of the sidebar (works from every page).
# on_click runs before the page is drawn, so the page is logged out at once.
if ss.get("user"):
    st.sidebar.divider()
    st.sidebar.caption(f"Logged in: **{ss.user['name']}** ({ss.user['role']})")
    st.sidebar.button("🔓 Logout", on_click=log_out, width="stretch", key="logout")

page_login()          # the only page of app4.py
