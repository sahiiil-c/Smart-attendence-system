"""
tests/test_face.py - run with:   python -m pytest tests -v

What is tested:
  1. the face model loads
  2. face checks: same person >= 0.5, different person lower, too dark / no face / many faces rejected
  3. attendance rules: no Time Out before Time In, cooldown, statuses
  4. storage: SQLite saves and reads back correctly
  5. auto capture rules: centred face, wait + head turn between photos, 2 frames then pause
  6. password hashing and admin edit / delete (CRUD)
  7. Admin / Teacher / Student logins and camera pages, with no data (no crash)
  8. the live camera loop: frames in -> photos taken and attendance marked without clicks
  9. database upgrade (separate student / teacher attendance, classes) and app2.py:
     sign up on the Login page, mark MY attendance with a face check, teacher sees class students

Your own photos (optional, recommended): put these in tests/photos/
    same_1.jpg   - you
    same_2.jpg   - you again (another day / angle)
    other.jpg    - a different person
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from assets import attendance_rules as rules
from assets import auth
from assets import auto_capture
from assets import face_engine
from assets.storage import SQLiteStore, make_next_id

PHOTOS = os.path.join(ROOT, "tests", "photos")


@pytest.fixture(scope="module")
def model():
    return face_engine.load_model()


def sample_image(name):
    """Test pictures that ship inside the insightface package."""
    import insightface
    path = os.path.join(os.path.dirname(insightface.__file__), "data", "images", name)
    return cv2.imread(path)


def embed(model, img):
    emb, message = face_engine.get_face_embedding(model, img)
    assert emb is not None, message
    return emb


# ---------------------------------------------------------------- 1. model

def test_model_loads(model):
    assert "detection" in model.models and "recognition" in model.models


# ---------------------------------------------------------------- 2. faces

def test_same_and_different_person_your_photos(model):
    files = [os.path.join(PHOTOS, f) for f in ["same_1.jpg", "same_2.jpg", "other.jpg"]]
    if not all(os.path.exists(f) for f in files):
        pytest.skip("put same_1.jpg, same_2.jpg and other.jpg in tests/photos/ to run this test")

    a1, a2, b = [embed(model, cv2.imread(f)) for f in files]

    same = float(a1 @ a2)
    different = float(a1 @ b)
    print(f"same person: {same:.3f}   different person: {different:.3f}")

    assert same >= face_engine.MATCH_THRESHOLD
    assert different < same
    assert different < face_engine.MATCH_THRESHOLD


def test_different_people_score_low(model):
    # t1.jpg is a group photo: every face is a different person
    faces = model.get(sample_image("t1.jpg"))
    assert len(faces) >= 2

    embs = np.array([f.normed_embedding for f in faces])
    scores = embs @ embs.T
    np.fill_diagonal(scores, -1)

    print(f"highest score between two different people: {scores.max():.3f}")
    assert scores.max() < face_engine.MATCH_THRESHOLD


def one_person_photo(model):
    """Cut one person (with a wide margin) out of the t1.jpg group photo."""
    group = sample_image("t1.jpg")
    face = model.get(group)[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    m = (x2 - x1)                                   # margin = one face width
    crop = group[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
    return cv2.resize(crop, None, fx=3, fy=3)       # make the face big enough


def test_same_photo_with_other_lighting_still_matches(model):
    img = one_person_photo(model)
    darker = cv2.convertScaleAbs(img, alpha=0.6, beta=0)     # 40% darker
    flipped = cv2.flip(img, 1)                               # mirror image

    original = embed(model, img)
    assert float(original @ embed(model, darker)) >= face_engine.MATCH_THRESHOLD
    assert float(original @ embed(model, flipped)) >= face_engine.MATCH_THRESHOLD


def test_best_match_picks_right_person(model):
    faces = model.get(sample_image("t1.jpg"))
    embs = np.array([f.normed_embedding for f in faces])
    ids = np.array([f"ID_{i:03}" for i in range(len(faces))])

    person_id, score = face_engine.best_match(embs[2], ids, embs)
    assert person_id == "ID_002" and score > 0.99

    assert face_engine.best_match(embs[0], np.array([]), np.zeros((0, 512))) == (None, 0.0)


def test_rejects_dark_empty_and_group_photos(model):
    img = sample_image("Tom_Hanks_54745.png")

    dark = (img * 0.1).astype(np.uint8)
    emb, message = face_engine.get_face_embedding(model, dark)
    assert emb is None and "dark" in message.lower()

    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    emb, message = face_engine.get_face_embedding(model, blank)
    assert emb is None and "no face" in message.lower()

    emb, message = face_engine.get_face_embedding(model, sample_image("t1.jpg"))
    assert emb is None and "more than one" in message.lower()


# ---------------------------------------------------------------- 3. rules

NOW = datetime(2026, 10, 28, 10, 0, 0)


def test_first_scan_is_time_in_never_time_out():
    assert rules.decide_scan(None, NOW) == ("time_in", 0)
    assert rules.decide_scan({"time_in": None, "time_out": None}, NOW)[0] == "time_in"


def test_cooldown_then_time_out_then_done():
    just_in = {"time_in": (NOW - timedelta(seconds=10)).strftime("%H:%M:%S"), "time_out": None}
    action, left = rules.decide_scan(just_in, NOW)
    assert action == "cooldown" and left == rules.COOLDOWN_SECONDS - 10

    long_ago = {"time_in": (NOW - timedelta(seconds=rules.COOLDOWN_SECONDS + 1)).strftime("%H:%M:%S"),
                "time_out": None}
    assert rules.decide_scan(long_ago, NOW)[0] == "time_out"

    finished = {"time_in": "09:00:00", "time_out": "09:30:00"}
    assert rules.decide_scan(finished, NOW)[0] == "done"


def test_status_rules():
    day = "2026-10-28"
    assert rules.compute_status(day, "09:00:00", "17:00:00", NOW) == "Present"
    assert rules.compute_status(day, "09:00:00", "", NOW) == "Pending"
    assert rules.compute_status(day, "09:00:00", "", NOW.replace(hour=18)) == "Absent"
    assert rules.compute_status(day, "", "", NOW) == "Pending"
    assert rules.compute_status(day, "", "", NOW.replace(hour=16)) == "Absent"


def test_report_lists_absent_people_and_handles_no_data():
    assert rules.build_report([], [], NOW).empty

    people = [{"id": "ID_001", "name": "A", "role": "student", "registered_on": "2026-10-01"},
              {"id": "ID_002", "name": "B", "role": "student", "registered_on": "2026-10-01"}]
    records = [{"person_id": "ID_001", "date": "2026-10-27", "time_in": "09:00:00", "time_out": "17:00:00"}]

    report = rules.build_report(people, records, NOW)
    assert list(report["Attendance"]) == ["Present", "Absent"]


# ---------------------------------------------------------------- 4. storage

def test_sqlite_store_roundtrip(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))

    assert store.next_person_id("student") == "ID_001"
    store.save_person("ID_001", "Test Person", "student", "2026-10-28")
    assert store.next_person_id("student") == "ID_002"
    assert store.next_person_id("teacher") == "T_001"
    assert store.get_person("id_001")["name"] == "Test Person"
    assert store.get_person("ID_0X1") is None

    embs = np.random.rand(3, 512).astype(np.float32)
    store.save_embeddings("ID_001", embs)
    ids, loaded = store.get_all_embeddings()
    assert list(ids) == ["ID_001"] * 3 and np.allclose(loaded, embs)

    store.mark_time_in("ID_001", "2026-10-28", "09:00:00")
    store.mark_time_out("ID_001", "2026-10-28", "17:00:00")
    assert store.get_record("ID_001", "2026-10-28")["time_out"] == "17:00:00"
    assert store.get_record("ID_001", "2026-10-29") is None

    assert make_next_id(["ID_001", "ID_007", "T_002"], "student") == "ID_008"


# ---------------------------------------------------------------- 5. auto capture rules

def test_registration_checks_centre(model):
    img = one_person_photo(model)
    face, _ = face_engine.get_face(model, img)
    assert face_engine.registration_checks(face, img) is None        # face in the middle

    h, w = img.shape[:2]
    wide = cv2.copyMakeBorder(img, 0, 0, 0, w * 2, cv2.BORDER_CONSTANT)  # face now far left
    face, _ = face_engine.get_face(model, wide)
    assert "center" in face_engine.registration_checks(face, wide).lower()


def test_auto_registration_needs_gap_and_head_turn(model):
    img = one_person_photo(model)
    first = embed(model, img)
    variant = embed(model, cv2.flip(cv2.convertScaleAbs(img, alpha=0.7, beta=0), 1))
    print(f"similarity first vs changed photo: {float(first @ variant):.3f}")

    # first photo: nothing to wait for
    assert auto_capture.auto_registration_ready(first, [], 0.0, 100.0) is None

    # same photo again right away -> wait; a bit later -> still the same pose -> "turn your head"
    assert auto_capture.auto_registration_ready(first, [first], 100.0, 100.5) == "Hold on..."
    assert "turn your head" in auto_capture.auto_registration_ready(first, [first], 100.0, 103.0).lower()

    # never stuck: after waiting long enough the same pose is accepted anyway
    late = 100.0 + auto_capture.CAPTURE_GAP + auto_capture.MAX_WAIT_FOR_CHANGE + 0.1
    assert auto_capture.auto_registration_ready(first, [first], 100.0, late) is None

    # a changed photo of the same person, after the gap -> accepted
    assert auto_capture.auto_registration_ready(variant, [first], 100.0, 103.0) is None
    ok, _, dup = auto_capture.check_registration_photo(variant, [first], np.array([]), np.zeros((0, 512)))
    assert ok and dup is None


def test_registration_blocks_duplicate_and_other_person(model):
    faces = model.get(sample_image("t1.jpg"))
    a, b = faces[0].normed_embedding, faces[1].normed_embedding

    ok, _, dup = auto_capture.check_registration_photo(a, [], np.array(["ID_001"]), np.array([a]))
    assert not ok and dup == "ID_001"

    ok, message, dup = auto_capture.check_registration_photo(b, [a], np.array([]), np.zeros((0, 512)))
    assert not ok and dup is None and "same person" in message


def test_auto_attendance_confirms_then_pauses():
    state = auto_capture.new_watch_state()

    assert auto_capture.attendance_step(state, None, 0.0) == "unknown"
    assert auto_capture.attendance_step(state, "ID_001", 1.0) == "confirming"
    assert auto_capture.attendance_step(state, "ID_002", 1.4) == "confirming"   # someone else: restart
    assert auto_capture.attendance_step(state, "ID_002", 1.8) == "mark"
    assert auto_capture.attendance_step(state, "ID_002", 2.2) == "paused"      # during the pause
    assert auto_capture.seconds_paused(state, 2.2) == 4

    after = 1.8 + auto_capture.ATTENDANCE_PAUSE + 0.1
    assert auto_capture.attendance_step(state, "ID_002", after) == "confirming"


# ---------------------------------------------------------------- 6. passwords + admin data

def test_password_hashing():
    stored = auth.hash_password("secret123")
    assert "secret123" not in stored
    assert stored != auth.hash_password("secret123")                # random salt
    assert auth.check_password("secret123", stored)
    assert not auth.check_password("wrong", stored)
    assert not auth.check_password("secret123", "garbage")


def test_admin_crud(tmp_path):
    store = SQLiteStore(str(tmp_path / "crud.db"))
    store.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    store.save_person("ID_001", "Riya S", "student", "2026-10-01")
    store.save_embeddings("ID_001", np.random.rand(2, 512))

    store.update_person("ID_001", "Riya Sharma", "student")
    assert store.get_person("ID_001")["name"] == "Riya Sharma"

    store.save_record("ID_001", "2026-10-28", "09:00:00", "")
    store.save_record("ID_001", "2026-10-28", "09:00:00", "17:00:00")     # edit = replace
    assert store.get_record("ID_001", "2026-10-28")["time_out"] == "17:00:00"
    store.delete_record("ID_001", "2026-10-28")
    assert store.get_record("ID_001", "2026-10-28") is None

    store.save_teacher_account(" Amit ", auth.hash_password("pass123"), "T_001")
    account = store.get_teacher_account("amit")
    assert account["person_id"] == "T_001" and auth.check_password("pass123", account["password_hash"])

    # deleting a person removes everything linked to them
    store.save_record("ID_001", "2026-10-28", "09:00:00", "")
    store.delete_person("ID_001")
    store.delete_person("T_001")
    ids, _ = store.get_all_embeddings()
    assert store.get_people() == [] and store.get_attendance() == []
    assert len(ids) == 0 and store.get_teacher_accounts() == []


# ---------------------------------------------------------------- 7. pages and logins, no data

@pytest.fixture
def empty_app(tmp_path, monkeypatch):
    return make_app("app.py", tmp_path, monkeypatch)


@pytest.fixture
def empty_app2(tmp_path, monkeypatch):
    return make_app("app2.py", tmp_path, monkeypatch)


def make_app(file_name, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    db_path = str(tmp_path / "empty.db")
    monkeypatch.setenv("ATTENDANCE_DB", db_path)
    monkeypatch.setenv("ATTENDANCE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.chdir(ROOT)

    # storage reads ATTENDANCE_DB when it is imported, so import it fresh,
    # and forget the store cached by an earlier test
    # (the assets package also keeps the old module as an attribute, so remove it there too)
    import streamlit as st
    import assets
    sys.modules.pop("assets.storage", None)
    vars(assets).pop("storage", None)
    st.cache_resource.clear()

    at = AppTest.from_file(os.path.join(ROOT, file_name), default_timeout=60)
    # own secrets, so the tests never use the real secrets.toml (no Supabase)
    at.secrets = {"ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "admin123"}
    at.db = SQLiteStore(db_path)
    return at


def open_login(at, role):
    at.run()
    at.sidebar.radio[0].set_value("🔐 Login").run()
    at.segmented_control[0].set_value(role).run()
    return at


def test_admin_login(empty_app):
    at = open_login(empty_app, "Admin")
    at.text_input[0].input("admin")
    at.text_input[1].input("wrong")
    at.button[0].click().run()
    assert any("Wrong" in e.value for e in at.error)

    at.text_input[1].input("admin123")
    at.button[0].click().run()

    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    assert any("No attendance records yet" in i.value for i in at.info)

    # Database tab: no Supabase secrets -> online switch disabled, offline data viewable
    assert at.segmented_control(key="db_mode").disabled
    assert any("Sync needs the online database" in i.value for i in at.info)
    assert any("This table is empty" in i.value for i in at.info)


def test_sync_between_two_stores(tmp_path):
    from assets.storage import sync_stores

    a = SQLiteStore(str(tmp_path / "a.db"))
    b = SQLiteStore(str(tmp_path / "b.db"))
    emb = np.ones(512, dtype=np.float32) / np.sqrt(512)

    a.save_person("ID_001", "Riya S", "student", "2026-10-01")
    a.save_embeddings("ID_001", [emb, emb])
    a.save_record("ID_001", "2026-10-02", "09:00:00", "17:00:00")
    a.save_teacher_account("riya", "hash", "ID_001")

    # same day on both sides: A is complete, B has only Time In
    a.save_person("ID_002", "Amit C", "student", "2026-10-01")
    b.save_person("ID_002", "Amit C", "student", "2026-10-01")
    a.save_record("ID_002", "2026-10-02", "09:00:00", "17:00:00")
    b.save_record("ID_002", "2026-10-02", "09:00:00", None)

    # same ID, different person -> conflict, not copied
    a.save_person("ID_003", "Neha P", "student", "2026-10-01")
    b.save_person("ID_003", "Someone Else", "student", "2026-10-01")
    a.save_record("ID_003", "2026-10-03", "09:00:00", None)

    # only in B -> stays (sync never deletes)
    b.save_person("ID_004", "Only B", "student", "2026-10-01")

    r = sync_stores(a, b)

    assert (r["people"], r["embeddings"], r["records"], r["accounts"]) == (1, 2, 2, 1)
    assert len(r["conflicts"]) == 1 and "ID_003" in r["conflicts"][0]
    assert b.get_person("ID_001")["name"] == "Riya S"
    assert b.get_record("ID_002", "2026-10-02")["time_out"] == "17:00:00"
    assert b.get_record("ID_003", "2026-10-03") is None
    assert b.get_person("ID_003")["name"] == "Someone Else"
    assert b.get_person("ID_004") is not None
    assert b.get_teacher_account("riya")["password_hash"] == "hash"

    # running it again copies nothing new
    r = sync_stores(a, b)
    assert (r["people"], r["embeddings"], r["records"], r["accounts"]) == (0, 0, 0, 0)


def test_storage_mode_setting(tmp_path, monkeypatch):
    from assets import storage

    monkeypatch.setattr(storage, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(storage, "online_available", lambda: True)

    assert storage.get_mode() == "offline"          # default without a settings file
    storage.set_mode("online")
    assert storage.get_mode() == "online"

    # online saved but no Supabase secrets -> falls back to offline
    monkeypatch.setattr(storage, "online_available", lambda: False)
    assert storage.get_mode() == "offline"


def test_admin_views_records_by_date(empty_app):
    at = empty_app

    # two recent working days (Mon-Fri) before today
    today = rules.now_local().date()
    days = [today - timedelta(days=n) for n in range(1, 10)]
    day1, day2 = [d for d in days if d.weekday() < 5][:2]

    at.db.save_person("ID_001", "Riya S", "student", (today - timedelta(days=15)).isoformat())
    at.db.save_record("ID_001", day1.isoformat(), "09:00:00", "17:00:00")
    at.db.save_record("ID_001", day2.isoformat(), "09:10:00", "17:05:00")

    at = open_login(at, "Admin")
    at.text_input[0].input("admin")
    at.text_input[1].input("admin123")
    at.button[0].click().run()

    # only day1 selected -> the table shows only day1
    at.date_input(key="view_from").set_value(day1).run()
    at.date_input(key="view_to").set_value(day1).run()

    assert not at.exception, at.exception
    table = [d.value for d in at.dataframe if "Date" in d.value.columns][0]
    assert list(table["Date"]) == [day1.isoformat()]

    # From after To -> warning instead of a table
    at.date_input(key="view_from").set_value(day1 + timedelta(days=1)).run()
    assert any("must be before" in w.value for w in at.warning)


def test_teacher_login_sees_no_admin_tabs(empty_app):
    at = empty_app
    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")

    at = open_login(at, "Teacher")
    at.text_input[0].input("amit")
    at.text_input[1].input("pass123")
    at.button[0].click().run()

    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["👨‍🏫 My attendance", "🎓 Students"]
    assert any("No attendance records yet" in i.value for i in at.info)


def test_student_login_with_no_data(empty_app):
    at = empty_app
    at.db.save_person("ID_001", "New Student", "student", "2026-10-28")

    at = open_login(at, "Student")
    at.text_input[0].input("id_001")
    at.button[0].click().run()

    assert not at.exception, at.exception
    assert any("No attendance records yet" in i.value for i in at.info)


@pytest.mark.parametrize("page", ["📝 Register Face", "📷 Mark Attendance"])
@pytest.mark.parametrize("mode", ["Auto", "Manual"])
def test_camera_pages_render(empty_app, page, mode):
    at = empty_app
    at.run()
    at.sidebar.radio[0].set_value(page).run()

    if page == "📝 Register Face":
        at.text_input[0].input("Test Person").run()

    at.segmented_control[0].set_value(mode).run()
    assert not at.exception, at.exception


# ---------------------------------------------------------------- 8. live camera loop (Python side)

def as_camera_frame(seq, img):
    """A frame exactly like the browser component sends it."""
    import base64
    jpeg = cv2.imencode(".jpg", img)[1].tobytes()
    return {"seq": seq, "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}


def send_frame(at, key, seq, img):
    at.session_state[f"{key}__frame"] = as_camera_frame(seq, img)
    at.run()
    assert not at.exception, at.exception


def test_auto_register_and_auto_attendance(empty_app, model):
    at = empty_app
    img = one_person_photo(model)
    changed = cv2.flip(cv2.convertScaleAbs(img, alpha=0.7, beta=0), 1)

    # ---- register: frames arrive, photos are taken without any click ----
    at.run()
    at.sidebar.radio[0].set_value("📝 Register Face").run()
    at.text_input[0].input("Auto Person").run()

    send_frame(at, "reg_live", 1, img)
    assert len(at.session_state.reg_embs) == 1 and at.session_state.reg_ack == 1

    send_frame(at, "reg_live", 2, img)                        # too soon / same pose
    assert len(at.session_state.reg_embs) == 1

    for seq in (3, 4):
        at.session_state.reg_last -= 10                       # pretend time passed
        send_frame(at, "reg_live", seq, changed if seq == 3 else img)

    assert len(at.session_state.reg_embs) == 3
    at.button[0].click().run()                                # Save Registration
    assert at.db.get_person("ID_001")["name"] == "Auto Person"

    # ---- attendance: 2 frames of the same face -> Time In, then a pause ----
    at.sidebar.radio[0].set_value("📷 Mark Attendance").run()
    send_frame(at, "att_live", 1, img)
    assert at.session_state.att_cam[0] == "Hold still..."
    send_frame(at, "att_live", 2, img)
    assert "Time In marked" in at.session_state.att_result[1]

    send_frame(at, "att_live", 3, img)                        # during the pause
    assert at.session_state.att_cam[0].startswith("Next person in")
    assert at.db.get_record("ID_001", rules.now_local().date().isoformat())["time_in"]


# ---------------------------------------------------------------- 9. upgrade + app2.py

def test_old_database_is_upgraded_without_losing_data(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")

    # a data.db in the OLD format: one attendance table, no year / division
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE people (id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
                                 registered_on TEXT NOT NULL);
            CREATE TABLE attendance (person_id TEXT NOT NULL, date TEXT NOT NULL, time_in TEXT,
                                     time_out TEXT, PRIMARY KEY (person_id, date));
            INSERT INTO people VALUES ('ID_001', 'Riya S', 'student', '2026-10-01'),
                                      ('T_001', 'Amit C', 'teacher', '2026-10-01');
            INSERT INTO attendance VALUES ('ID_001', '2026-10-02', '09:00:00', '17:00:00'),
                                          ('T_001', '2026-10-02', '08:50:00', NULL);
        """)

    store = SQLiteStore(path)

    assert [r["person_id"] for r in store.get_attendance("student")] == ["ID_001"]
    assert [r["person_id"] for r in store.get_attendance("teacher")] == ["T_001"]
    assert store.get_person("ID_001")["year"] is None
    assert os.path.exists(str(tmp_path / "old_backup_before_upgrade.db"))

    with sqlite3.connect(path) as db:          # the old rows are still there, renamed
        assert db.execute("SELECT COUNT(*) FROM attendance_backup").fetchone()[0] == 2

    SQLiteStore(path)                          # opening again changes nothing
    assert len(store.get_attendance()) == 2


def test_classes_and_role_change(tmp_path):
    from assets.storage import students_of_teacher

    store = SQLiteStore(str(tmp_path / "c.db"))
    store.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    store.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    store.save_person("ID_002", "Neha P", "student", "2026-10-01", "TE", "B")
    store.set_teacher_classes("T_001", [("SE", "A")])

    assert [s["id"] for s in students_of_teacher(store, "T_001")] == ["ID_001"]

    store.mark_time_in("ID_002", "2026-10-02", "09:00:00")
    assert len(store.get_attendance("student")) == 1

    # ID_002 becomes a teacher -> the record moves to teacher_attendance
    store.update_person("ID_002", "Neha P", "teacher")
    assert store.get_attendance("student") == []
    assert store.get_record("ID_002", "2026-10-02")["time_in"] == "09:00:00"


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_app2_signup_then_mark_my_attendance(empty_app2, model):
    at = empty_app2
    img = one_person_photo(model)
    changed = cv2.flip(cv2.convertScaleAbs(img, alpha=0.7, beta=0), 1)

    # ---- sign up on the Login page ----
    at.run()
    at.segmented_control(key="login_action").set_value("Sign up").run()
    at.selectbox(key="signup_year").set_value("SE")
    at.selectbox(key="signup_div").set_value("B")
    at.text_input(key="signup_name").input("Riya Sharma").run()

    send_frame(at, "reg_live", 1, img)
    for seq in (2, 3):
        at.session_state.reg_last -= 10                       # pretend time passed
        send_frame(at, "reg_live", seq, changed if seq == 2 else img)

    button(at, "✅ Save and log in").click().run()
    assert not at.exception, at.exception
    assert at.session_state.user["id"] == "ID_001"
    person = at.db.get_person("ID_001")
    assert (person["year"], person["division"]) == ("SE", "B")

    # ---- logged in: Mark my attendance is shown first ----
    send_frame(at, "me_live", 1, img)
    assert at.session_state.me_cam[0] == "Hold still..."
    send_frame(at, "me_live", 2, img)
    assert "Time In marked" in at.session_state.me_result[1]
    assert len(at.db.get_attendance("student")) == 1
    assert at.db.get_attendance("teacher") == []


def logged_in_user(at):
    """The logged-in user, or None (the key does not exist before the first login)."""
    return at.session_state["user"] if "user" in at.session_state else None


def student_id_step(at, number):
    """Log in as Student, step 1: type the number after "ID_" -> the face check starts."""
    at.run()
    at.text_input(key="login_number").input(number)
    button(at, "Continue").click().run()
    assert not at.exception, at.exception


def test_app2_student_logs_in_with_own_face(empty_app2, model):
    at = empty_app2
    img = one_person_photo(model)
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    at.db.save_embeddings("ID_001", [embed(model, img)])

    student_id_step(at, "1")
    assert not logged_in_user(at)                            # not yet: face first
    assert at.session_state.login_pending["id"] == "ID_001"

    send_frame(at, "login_live", 1, img)
    assert at.session_state.login_face_cam[0] == "Hold still..."
    send_frame(at, "login_live", 2, img)                      # face confirmed -> logged in
    assert at.session_state.user["id"] == "ID_001"
    assert at.session_state.login_token                      # remembered for a reload
    assert auth.user_from_token(at.db, at.session_state.login_token)["id"] == "ID_001"

    # logged in: Mark my attendance
    send_frame(at, "me_live", 1, img)
    send_frame(at, "me_live", 2, img)
    assert "Time In marked" in at.session_state.me_result[1]

    # Logout forgets the login in the database too
    token = at.session_state.login_token
    assert at.sidebar.button[-1].label == "🔓 Logout"          # at the end of the sidebar
    at.sidebar.button[-1].click().run()
    assert not logged_in_user(at)
    assert auth.user_from_token(at.db, token) is None


def test_app2_rejects_someone_elses_face(empty_app2, model):
    at = empty_app2
    faces = model.get(sample_image("t1.jpg"))
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    at.db.save_embeddings("ID_001", [faces[1].normed_embedding])   # a different person

    student_id_step(at, "001")

    img = one_person_photo(model)                                   # face of faces[0]
    send_frame(at, "login_live", 1, img)
    send_frame(at, "login_live", 2, img)

    assert at.session_state.login_face_cam[0] == "This face is not Riya S"
    assert not logged_in_user(at)                                   # stays logged out
    assert at.db.get_attendance() == []


def test_app2_unknown_student_id(empty_app2):
    at = empty_app2
    student_id_step(at, "999")
    assert any("Invalid Student ID" in e.value for e in at.error)


def test_app2_teacher_sees_only_class_students(empty_app2):
    at = empty_app2
    today = rules.now_local().date().isoformat()
    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    at.db.save_person("ID_002", "Neha P", "student", "2026-10-01", "TE", "B")
    at.db.set_teacher_classes("T_001", [("SE", "A")])
    at.db.save_record("ID_001", today, "09:00:00", None)
    at.db.save_record("ID_002", today, "09:00:00", None)

    at.run()
    at.segmented_control(key="login_role").set_value("Teacher").run()

    # wrong password
    at.text_input[0].input("amit")
    at.text_input[1].input("wrong1")
    button(at, "Login").click().run()
    assert any("Wrong username or password" in e.value for e in at.error)

    at.text_input[1].input("pass123")
    button(at, "Login").click().run()
    assert at.session_state.user["id"] == "T_001"
    at.segmented_control(key="teacher_view").set_value("🎓 My students").run()

    assert not at.exception, at.exception
    table = [d.value for d in at.dataframe if "Class" in d.value.columns][0]
    assert set(table["ID"]) == {"ID_001"}
    assert set(table["Class"]) == {"SE-A"}


def test_app2_teacher_signup_checks_username_and_password(empty_app2):
    at = empty_app2
    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")

    at.run()
    at.segmented_control(key="login_action").set_value("Sign up").run()
    at.segmented_control(key="signup_role").set_value("teacher").run()
    at.text_input(key="signup_name").input("Neha Patil")

    def try_login(user, password, confirm):
        at.text_input(key="signup_user").input(user)
        at.text_input(key="signup_pass").input(password)
        at.text_input(key="signup_pass2").input(confirm).run()
        assert not at.exception, at.exception
        return [w.value for w in at.warning]

    assert "This username is already taken." in try_login("Amit", "secret1", "secret1")
    assert "The password needs at least 6 characters." in try_login("neha", "abc", "abc")
    assert "The two passwords are not the same." in try_login("neha", "secret1", "secret2")
    assert try_login("neha", "secret1", "secret1") == []         # ok -> identity check starts
    assert at.segmented_control(key="id_mode")                     # identity check camera shown


def test_login_sessions(tmp_path):
    from datetime import timedelta as td

    store = SQLiteStore(str(tmp_path / "s.db"))
    store.save_person("ID_001", "Riya S", "student", "2026-10-01")

    token = auth.start_login(store, {"role": "Student", "id": "ID_001", "name": "Riya S"})
    assert auth.user_from_token(store, token)["id"] == "ID_001"
    assert auth.user_from_token(store, "made-up-token") is None
    assert store.get_login_session(token) is None                 # only the hash is stored

    admin = auth.start_login(store, {"role": "Admin", "id": "ADMIN", "name": "Admin"})
    assert auth.user_from_token(store, admin)["role"] == "Admin"

    # expired
    old = auth.start_login(store, {"role": "Student", "id": "ID_001", "name": "Riya S"})
    expired = (datetime.now(timezone.utc) - td(days=1)).isoformat()
    with store._connect() as db:
        db.execute("UPDATE login_sessions SET expires_at = ? WHERE token_hash = ?",
                   (expired, auth._token_hash(old)))
    assert auth.user_from_token(store, old) is None

    # logout, and deleting the person, end the login
    auth.end_login(store, admin)
    assert auth.user_from_token(store, admin) is None
    store.delete_person("ID_001")
    assert auth.user_from_token(store, token) is None


def test_app2_admin_logs_in_through_the_teacher_form(empty_app2):
    at = empty_app2
    at.run()

    # no separate Admin option any more
    assert at.segmented_control(key="login_role").options == ["Teacher", "Student"]

    at.segmented_control(key="login_role").set_value("Teacher").run()
    at.text_input[0].input("admin")
    at.text_input[1].input("admin123")
    button(at, "Login").click().run()

    assert not at.exception, at.exception
    assert logged_in_user(at)["role"] == "Admin"
    assert "🗄 Database" in [t.label for t in at.tabs]


def test_teachers_cannot_take_the_admin_username(empty_app2):
    at = empty_app2
    at.run()
    at.segmented_control(key="login_action").set_value("Sign up").run()
    at.segmented_control(key="signup_role").set_value("teacher").run()
    at.text_input(key="signup_name").input("Neha Patil")
    at.text_input(key="signup_user").input("Admin")
    at.text_input(key="signup_pass").input("secret1")
    at.text_input(key="signup_pass2").input("secret1").run()

    assert "This username is already taken." in [w.value for w in at.warning]



def test_one_account_per_teacher(tmp_path):
    import sqlite3
    store = SQLiteStore(str(tmp_path / "t.db"))
    store.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    store.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")

    assert store.get_account_of_teacher("T_001")["username"] == "amit"
    assert store.get_account_of_teacher("T_002") is None

    # a second account for the same teacher is refused by the database
    with pytest.raises(sqlite3.IntegrityError):
        store.save_teacher_account("amit2", auth.hash_password("pass456"), "T_001")
    assert [a["username"] for a in store.get_teacher_accounts()] == ["amit"]

    # reset password (same username) still works
    store.save_teacher_account("amit", auth.hash_password("newpass1"), "T_001")
    assert auth.check_password("newpass1", store.get_teacher_account("amit")["password_hash"])


def test_sync_skips_a_second_account_for_a_teacher(tmp_path):
    from assets.storage import sync_stores

    a = SQLiteStore(str(tmp_path / "a.db"))
    b = SQLiteStore(str(tmp_path / "b.db"))
    for store in (a, b):
        store.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    a.save_teacher_account("amit_a", "hash-a", "T_001")
    b.save_teacher_account("amit_b", "hash-b", "T_001")

    r = sync_stores(a, b)

    assert r["accounts"] == 0
    assert any("T_001 already has an account" in c for c in r["conflicts"])
    assert [x["username"] for x in b.get_teacher_accounts()] == ["amit_b"]


def test_admin_add_account_lists_only_teachers_without_one(empty_app2):
    at = empty_app2
    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.save_person("T_002", "Neha P", "teacher", "2026-10-01")
    at.db.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")

    at.run()
    at.segmented_control(key="login_role").set_value("Teacher").run()
    at.text_input[0].input("admin")
    at.text_input[1].input("admin123")
    button(at, "Login").click().run()

    assert not at.exception, at.exception
    options = at.selectbox(key="acc_teacher").options
    assert len(options) == 1 and "T_002" in options[0]           # T_001 already has one


# ---------------------------------------------------------------- 10. admin app settings

def test_settings_keep_each_other(tmp_path, monkeypatch):
    """The storage mode stays in settings.json, every other setting goes to the database."""
    import json

    from assets import storage

    # defaults
    assert storage.get_capture_default() == "Auto"
    assert storage.get_cooldown("student") == rules.COOLDOWN_SECONDS
    assert storage.get_cooldown("teacher") == rules.COOLDOWN_SECONDS

    storage.set_mode("offline")
    storage.save_setting("capture_mode", "Manual")
    storage.save_setting("cooldown_student", 120)
    storage.save_setting("cooldown_teacher", 600)

    assert storage.get_mode() == "offline"              # not wiped by the other settings
    assert storage.get_capture_default() == "Manual"
    assert storage.get_cooldown("student") == 120
    assert storage.get_cooldown("teacher") == 600

    # where they are kept
    assert json.load(open(storage.SETTINGS_PATH, encoding="utf-8")) == {"storage": "offline"}
    in_db = SQLiteStore(storage.DB_PATH).get_settings()
    assert (in_db["capture_mode"], in_db["cooldown_student"], in_db["cooldown_teacher"]) == \
        ("Manual", 120, 600)


def test_old_settings_json_values_are_still_read(tmp_path):
    """Values saved in settings.json before this change still work until saved again."""
    import json

    from assets import storage

    with open(storage.SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({"storage": "offline", "capture_mode": "Manual", "cooldown_teacher": 300}, f)

    assert storage.get_capture_default() == "Manual"
    assert storage.get_cooldown("teacher") == 300

    # copied into the database once (so Upload takes them online); "storage" stays in the file
    in_db = SQLiteStore(storage.DB_PATH).get_settings()
    assert in_db == {"capture_mode": "Manual", "cooldown_teacher": 300}

    storage.save_setting("cooldown_teacher", 90)          # changed in the database
    assert storage.get_cooldown("teacher") == 90


def test_cooldown_per_role_and_wait_text():
    just_in = {"time_in": (NOW - timedelta(minutes=3)).strftime("%H:%M:%S"), "time_out": None}

    # student cooldown 2 min -> 3 min after Time In: Time Out
    assert rules.decide_scan(just_in, NOW, cooldown=120)[0] == "time_out"

    # teacher cooldown 10 min -> still waiting 7 more minutes
    action, left = rules.decide_scan(just_in, NOW, cooldown=600)
    assert action == "cooldown" and left == 420
    assert rules.wait_text(left) == "7 min"

    assert rules.wait_text(45) == "45 seconds"
    assert rules.wait_text(200) == "3 min 20 s"


def test_admin_app_settings_tab(empty_app2, tmp_path, monkeypatch):
    import importlib
    import json

    at = empty_app2
    at.run()
    at.segmented_control(key="login_role").set_value("Teacher").run()
    at.text_input[0].input("admin")
    at.text_input[1].input("admin123")
    button(at, "Login").click().run()
    assert "⚙️ App settings" in [t.label for t in at.tabs]

    at.segmented_control(key="set_capture").set_value("Manual")
    at.number_input(key="set_cd_student").set_value(2.0)
    at.number_input(key="set_cd_teacher").set_value(10.0)
    button(at, "💾 Save settings").click().run()

    assert not at.exception, at.exception
    assert any("Settings saved" in s.value for s in at.success)

    saved = at.db.get_settings()                            # saved in the database
    assert saved["capture_mode"] == "Manual"
    assert (saved["cooldown_student"], saved["cooldown_teacher"]) == (120, 600)

    storage = importlib.import_module("assets.storage")   # the copy the app uses
    assert storage.get_cooldown("teacher") == 600

    # a new visitor: the camera switch starts on Manual
    fresh = make_app("app2.py", tmp_path, monkeypatch)
    fresh.run()
    fresh.segmented_control(key="login_action").set_value("Sign up").run()
    fresh.text_input(key="signup_name").input("Riya Sharma").run()
    assert fresh.segmented_control(key="reg_mode").value == "Manual"



# ---------------------------------------------------------------- 11. teacher sign-up identity check

def test_freeze_and_alerts_storage(tmp_path):
    store = SQLiteStore(str(tmp_path / "f.db"))
    store.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    token = auth.start_login(store, {"role": "Student", "id": "ID_001", "name": "Riya S"})

    assert store.get_person("ID_001")["frozen"] == 0
    store.set_frozen("ID_001", True)
    store.delete_login_sessions_of("ID_001")
    store.add_alert("ID_001", "Tried to sign up as teacher", "2026-10-28 10:00:00")

    assert store.get_person("ID_001")["frozen"] == 1
    assert auth.user_from_token(store, token) is None          # logged out everywhere
    [alert] = store.get_alerts("open")
    assert alert["student_id"] == "ID_001"

    store.set_alert_status(alert["id"], "restored")
    store.set_frozen("ID_001", False)
    assert store.get_alerts("open") == [] and len(store.get_alerts()) == 1
    assert store.get_person("ID_001")["frozen"] == 0


def start_teacher_signup(at, username="neha"):
    at.run()
    at.segmented_control(key="login_action").set_value("Sign up").run()
    at.segmented_control(key="signup_role").set_value("teacher").run()
    at.text_input(key="signup_name").input("Neha Patil")
    at.text_input(key="signup_user").input(username)
    at.text_input(key="signup_pass").input("secret1")
    at.text_input(key="signup_pass2").input("secret1").run()
    assert not at.exception, at.exception


def test_student_face_in_teacher_signup_freezes_the_student(empty_app2, model):
    at = empty_app2
    img = one_person_photo(model)
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    at.db.save_embeddings("ID_001", [embed(model, img)])

    start_teacher_signup(at)
    send_frame(at, "idcheck_live", 1, img)                 # ONE scan is enough

    # cancelled -> back on "Log in" with the message, nothing saved
    assert at.segmented_control(key="login_action").value == "Log in"
    assert any("Registration cancelled" in e.value for e in at.error)
    assert [p["id"] for p in at.db.get_people()] == ["ID_001"]
    assert at.db.get_teacher_account("neha") is None

    # the student is frozen and the admin has an alert
    assert at.db.get_person("ID_001")["frozen"] == 1
    [alert] = at.db.get_alerts("open")
    assert "Neha Patil" in alert["message"] and "neha" in alert["message"]

    # the frozen student can't log in
    at.segmented_control(key="login_role").set_value("Student").run()
    at.text_input[0].input("ID_001")
    button(at, "Continue").click().run()
    assert any("frozen" in e.value for e in at.error)
    pending = at.session_state["login_pending"] if "login_pending" in at.session_state else None
    assert not pending                                          # no face step for a frozen account


def login_admin(at):
    at.run()
    at.segmented_control(key="login_role").set_value("Teacher").run()
    at.text_input[0].input("admin")
    at.text_input[1].input("admin123")
    button(at, "Login").click().run()
    assert not at.exception, at.exception


def test_admin_restores_or_deletes_a_frozen_student(empty_app2):
    at = empty_app2
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    at.db.save_person("ID_002", "Neha P", "student", "2026-10-01", "SE", "A")
    for sid in ("ID_001", "ID_002"):
        at.db.set_frozen(sid, True)
        at.db.add_alert(sid, "Tried to sign up as teacher", "2026-10-28 10:00:00")

    login_admin(at)
    assert any("2 frozen account" in e.value for e in at.error)
    assert "🚨 Alerts (2)" in [t.label for t in at.tabs]

    alert_1 = next(a for a in at.db.get_alerts("open") if a["student_id"] == "ID_001")
    alert_2 = next(a for a in at.db.get_alerts("open") if a["student_id"] == "ID_002")

    at.button(key=f"alert_restore_{alert_1['id']}").click().run()
    assert at.db.get_person("ID_001")["frozen"] == 0

    at.checkbox(key=f"alert_del_ok_{alert_2['id']}").check().run()
    at.button(key=f"alert_del_{alert_2['id']}").click().run()
    assert at.db.get_person("ID_002") is None

    assert at.db.get_alerts("open") == []
    assert {a["status"] for a in at.db.get_alerts()} == {"restored", "deleted"}


def test_teacher_signup_passes_identity_check_with_a_new_face(empty_app2, model):
    at = empty_app2
    img = one_person_photo(model)

    start_teacher_signup(at)
    for seq in (1, 2):
        send_frame(at, "idcheck_live", seq, img)
        assert at.session_state.id_cam[0] == f"🛡 Scan {seq} / 3 – hold still"

    send_frame(at, "idcheck_live", 3, img)
    assert at.session_state.id_checked
    assert at.segmented_control(key="reg_mode")                  # photo taking starts now



@pytest.mark.parametrize("typed, expected", [
    ("1", "ID_001"), ("001", "ID_001"), (" 01 ", "ID_001"), ("id_001", "ID_001"), ("12", "ID_012"),
])
def test_app2_student_types_only_the_number(empty_app2, typed, expected):
    at = empty_app2
    at.db.save_person(expected, "Riya S", "student", "2026-10-01", "SE", "A")

    student_id_step(at, typed)

    assert at.session_state["login_pending"]["id"] == expected   # face check for this ID


@pytest.mark.parametrize("typed", ["abc", "", "ID_"])
def test_app2_student_number_must_be_a_number(empty_app2, typed):
    at = empty_app2
    student_id_step(at, typed)
    assert any("Enter the number of your Student ID" in e.value for e in at.error)


def test_capture_switch_can_be_hidden(empty_app2, tmp_path, monkeypatch):
    from assets import storage as st_mod
    monkeypatch.setattr(st_mod, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    assert st_mod.get_capture_switch_visible()                   # default: shown

    at = empty_app2
    login_admin(at)
    at.segmented_control(key="set_capture").set_value("Manual")
    at.toggle(key="set_capture_switch").set_value(False)
    button(at, "💾 Save settings").click().run()
    assert not at.exception, at.exception

    # a new visitor: no switch at all, the camera is Manual (Take photo)
    fresh = make_app("app2.py", tmp_path, monkeypatch)
    fresh.run()
    fresh.segmented_control(key="login_action").set_value("Sign up").run()
    fresh.text_input(key="signup_name").input("Riya Sharma").run()

    assert "reg_mode" not in [s.key for s in fresh.segmented_control]
    assert len(fresh.get("camera_input")) == 1

    # admin (still logged in) shows it again -> the switch is back
    at.toggle(key="set_capture_switch").set_value(True)
    button(at, "💾 Save settings").click().run()

    again = make_app("app2.py", tmp_path, monkeypatch)
    again.run()
    again.segmented_control(key="login_action").set_value("Sign up").run()
    again.text_input(key="signup_name").input("Riya Sharma").run()
    assert again.segmented_control(key="reg_mode").value == "Manual"


# ---------------------------------------------------------------- 12. late camera frames (no crash)

def test_late_frames_and_broken_frames_are_ignored():
    from streamlit.testing.v1 import AppTest

    def script():
        import types

        import streamlit as st

        from assets import live_camera

        ss = st.session_state

        # the camera left the page: its key is gone -> nothing happens
        live_camera._remember_frame("me_live")

        # value without a frame
        ss["cam_a"] = types.SimpleNamespace(frame=None)
        live_camera._remember_frame("cam_a")
        ss["cam_b"] = {"something": 1}
        live_camera._remember_frame("cam_b")

        # broken frames -> None
        ss["cam_c__frame"] = {"seq": 1, "image": "data:image/jpeg;base64,not-an-image"}
        ss["cam_d__frame"] = {"seq": 1}
        ss["results"] = [live_camera.take_frame("cam_c"), live_camera.take_frame("cam_d"),
                         live_camera.take_frame("never_sent")]

    at = AppTest.from_function(script)
    at.run()

    assert not at.exception, at.exception
    for key in ("me_live__frame", "cam_a__frame", "cam_b__frame"):
        assert key not in at.session_state
    assert at.session_state["results"] == [None, None, None]


def test_app2_late_frame_after_leaving_mark_my_attendance(empty_app2, model):
    at = empty_app2
    img = one_person_photo(model)
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    at.db.save_embeddings("ID_001", [embed(model, img)])

    # log in (number + face)
    student_id_step(at, "1")
    send_frame(at, "login_live", 1, img)
    send_frame(at, "login_live", 2, img)
    assert logged_in_user(at)["id"] == "ID_001"

    # leave Mark my attendance, then late frames arrive -> nothing happens
    at.segmented_control(key="student_view").set_value("📊 My attendance").run()
    send_frame(at, "me_live", 1, img)
    send_frame(at, "me_live", 2, img)
    assert at.db.get_attendance() == []

    # after Logout a late frame also does nothing
    at.sidebar.button[-1].click().run()
    send_frame(at, "me_live", 3, img)
    send_frame(at, "login_live", 3, img)
    assert not logged_in_user(at)
    assert at.db.get_attendance() == []
