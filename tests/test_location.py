"""
tests/test_location.py - the anti-fake-GPS / Wi-Fi checks of app4.py, with SIMULATED readings.

  python -m pytest tests/test_location.py -v

No real GPS is needed: readings are dicts {"lat", "lon", "accuracy", "time"} and the app is
fed with them the same way the browser component would send them.
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assets import attendance_rules as rules          # noqa: E402
from assets import auth                               # noqa: E402
from assets import location_checks as checks          # noqa: E402
from test_face import (button, embed, logged_in_user, make_app, model,  # noqa: E402,F401
                       one_person_photo, send_frame)

COLLEGE = (19.076000, 72.877700)        # a made-up college centre
HOME = (19.200000, 72.950000)           # ~16 km away


def real_readings(lat, lon, start=None):
    """3 readings like a real phone: tiny wobble, different accuracies, 2 s apart."""
    t = start or time.time()
    return [{"lat": lat + 0.000003 * i, "lon": lon - 0.000002 * i,
             "accuracy": 12.4 + 3.1 * i, "time": t + 2 * i} for i in range(3)]


def fake_readings(lat, lon, accuracy=5.0):
    """3 readings like a fake-GPS app: exactly the same every time."""
    t = time.time()
    return [{"lat": lat, "lon": lon, "accuracy": accuracy, "time": t + 2 * i} for i in range(3)]


# ====================================================================== 1. pure checks

def test_distance_and_geofence():
    assert checks.distance_m(*COLLEGE, *COLLEGE) == 0
    assert 15_000 < checks.distance_m(*COLLEGE, *HOME) < 17_000
    assert checks.inside_geofence(19.0765, 72.8777, *COLLEGE, 200)       # ~55 m away
    assert not checks.inside_geofence(*HOME, *COLLEGE, 200)


def test_average_position_is_used():
    readings = [{"lat": 10.0, "lon": 20.0}, {"lat": 10.2, "lon": 20.2}, {"lat": 10.4, "lon": 20.4}]
    assert checks.average_position(readings) == pytest.approx((10.2, 20.2))


def test_too_perfect_and_round_accuracy():
    assert checks.too_perfect(fake_readings(*COLLEGE))
    assert not checks.too_perfect(real_readings(*COLLEGE))

    assert checks.round_accuracy(fake_readings(*COLLEGE, accuracy=5))
    assert checks.round_accuracy(fake_readings(*COLLEGE, accuracy=10))
    assert not checks.round_accuracy(real_readings(*COLLEGE))

    # round on only SOME readings is normal
    mixed = real_readings(*COLLEGE)
    mixed[0]["accuracy"] = 5
    assert not checks.round_accuracy(mixed)


def test_impossible_travel():
    now = time.time()
    # home -> college (~16 km) in 10 minutes = ~96 km/h -> possible
    assert not checks.impossible_travel({"lat": HOME[0], "lon": HOME[1], "time": now - 600},
                                        *COLLEGE, now)
    # ... in 2 minutes = ~480 km/h -> impossible
    assert checks.impossible_travel({"lat": HOME[0], "lon": HOME[1], "time": now - 120},
                                    *COLLEGE, now)
    assert not checks.impossible_travel(None, *COLLEGE, now)               # first time ever


def test_same_spot():
    now = time.time()
    recent = [
        {"person_id": "ID_002", "lat": 19.0760004, "lon": 72.8777001, "time": now - 300},  # same 6 dp
        {"person_id": "ID_003", "lat": 19.076010, "lon": 72.877700, "time": now - 300},    # ~1 m away
        {"person_id": "ID_004", "lat": 19.076000, "lon": 72.877700, "time": now - 900},    # too long ago
        {"person_id": "ID_001", "lat": 19.076000, "lon": 72.877700, "time": now - 60},     # myself
    ]
    assert checks.same_spot("ID_001", 19.076000, 72.877700, now, recent) == ["ID_002"]


def test_client_ip_header_order():
    assert checks.client_ip({"CF-Connecting-IP": "1.1.1.1", "X-Forwarded-For": "2.2.2.2"}, "3.3.3.3") == "1.1.1.1"
    assert checks.client_ip({"X-Forwarded-For": "2.2.2.2, 10.0.0.1"}, "3.3.3.3") == "2.2.2.2"
    assert checks.client_ip({"x-forwarded-for": "2.2.2.2"}, "3.3.3.3") == "2.2.2.2"   # any case
    assert checks.client_ip({}, "3.3.3.3") == "3.3.3.3"


def test_ip_allowed_with_ips_and_ranges():
    allowed = ["203.0.113.7", "198.51.100.0/24", "not-an-ip"]
    assert checks.ip_allowed("203.0.113.7", allowed)
    assert checks.ip_allowed("198.51.100.250", allowed)
    assert not checks.ip_allowed("198.51.101.1", allowed)
    assert not checks.ip_allowed("garbage", allowed)


def test_decision():
    assert checks.decide(False, True, False, True, [])[0] == "blocked"     # outside geofence
    assert checks.decide(True, False, False, True, [])[0] == "blocked"     # face not recognised
    assert checks.decide(True, True, True, False, [])[0] == "blocked"      # Wi-Fi required, wrong IP
    assert checks.decide(True, True, False, False, []) == ("ok", [])       # Wi-Fi not required
    kind, reasons = checks.decide(True, True, False, True, [checks.FLAG_TOO_PERFECT])
    assert kind == "flagged" and reasons == [checks.FLAG_TOO_PERFECT]


def test_rejected_counts_as_absent_and_flagged_shows():
    now = rules.now_local()
    day = now.date().isoformat()
    people = [{"id": "ID_001", "name": "A", "role": "student", "registered_on": day}]
    base = {"person_id": "ID_001", "date": day, "time_in": "09:00:00", "time_out": "09:30:00"}

    def status(**extra):
        return rules.build_report(people, [{**base, **extra}], now)["Attendance"][0]

    assert status() == "Present"
    assert status(flagged=1, review="") == "Flagged"
    assert status(flagged=1, review="approved") == "Present"
    assert status(flagged=1, review="rejected") == "Absent"


# ====================================================================== 2. the app (app4.py)

@pytest.fixture
def app4(tmp_path, monkeypatch):
    """app4.py on an empty database, college at COLLEGE, Wi-Fi 203.0.113.0/24 (off by default)."""
    at = make_app("app4.py", tmp_path, monkeypatch)
    at.settings_file = tmp_path / "settings.json"
    write_settings(at)
    at.ip = "203.0.113.50"
    monkeypatch.setattr(checks, "client_ip", lambda headers, fallback=None: at.ip)
    return at


def write_settings(at, **extra):
    settings = {"college_lat": COLLEGE[0], "college_lon": COLLEGE[1], "geofence_radius_m": 200,
                "allowed_ips": ["203.0.113.0/24"], "wifi_required": False, **extra}
    at.settings_file.write_text(json.dumps(settings), encoding="utf-8")


def send_readings(at, key, readings):
    """Answer the location component's current request, like the browser would."""
    request = at.session_state[f"{key}_state"]["request"]
    at.session_state[f"{key}__readings"] = {"request": request, "readings": readings}
    at.run()
    assert not at.exception, at.exception


def student_login_location(at, number, readings):
    at.run()
    at.text_input(key="login_number").input(number)
    button(at, "Continue").click().run()
    send_readings(at, "login_loc", readings)


def add_student(at, model, sid="ID_001", name="Riya S"):
    img = one_person_photo(model)
    at.db.save_person(sid, name, "student", "2026-10-01", "SE", "A")
    at.db.save_embeddings(sid, [embed(model, img)])
    return img


def test_login_blocked_outside_geofence(app4):
    at = app4
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")

    student_login_location(at, "1", real_readings(*HOME))

    assert any("outside the college area" in e.value for e in at.error)
    assert "login_live" not in str(at.session_state)            # face step never opened
    assert not logged_in_user(at)


def test_login_blocked_without_college_wifi(app4):
    at = app4
    write_settings(at, wifi_required=True)
    at.ip = "8.8.8.8"                                            # mobile data, not college Wi-Fi
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")

    student_login_location(at, "1", real_readings(*COLLEGE))

    assert any("college Wi-Fi" in e.value for e in at.error)


def test_login_blocked_when_college_location_not_set(app4):
    at = app4
    at.settings_file.write_text("{}", encoding="utf-8")
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")

    student_login_location(at, "1", real_readings(*COLLEGE))

    assert any("college location is not set" in e.value for e in at.error)


def login_and_mark(at, img, readings_login, readings_mark):
    """Student: number -> location -> face -> Mark my attendance -> location -> face."""
    student_login_location(at, "1", readings_login)
    send_frame(at, "login_live", 1, img)
    send_frame(at, "login_live", 2, img)
    assert logged_in_user(at)["id"] == "ID_001"

    send_readings(at, "me_loc", readings_mark)
    send_frame(at, "me_live", 1, img)
    send_frame(at, "me_live", 2, img)
    return at.session_state["me_result"]


def test_honest_student_is_present_with_evidence_saved(app4, model):
    at = app4
    img = add_student(at, model)

    kind, text = login_and_mark(at, img, real_readings(*COLLEGE), real_readings(*COLLEGE))

    assert kind == "success" and "Time In marked" in text
    record = at.db.get_attendance("student")[0]
    assert record["flagged"] == 0 and not record["flag_reason"]
    assert record["ip_address"] == "203.0.113.50"
    assert len(json.loads(record["gps_readings"])["readings"]) == 3


def test_fake_gps_is_saved_as_flagged_then_teacher_reviews(app4, model):
    at = app4
    img = add_student(at, model)

    kind, text = login_and_mark(at, img, real_readings(*COLLEGE), fake_readings(*COLLEGE))

    assert kind == "warning" and "flagged" in text               # saved, but flagged
    record = at.db.get_attendance("student")[0]
    assert record["flagged"] == 1
    assert checks.FLAG_TOO_PERFECT in record["flag_reason"]
    assert checks.FLAG_ROUND_ACCURACY in record["flag_reason"]
    assert rules.build_report(at.db.get_people(), at.db.get_attendance(),
                              rules.now_local())["Attendance"][0] == "Flagged"

    # ---- the class teacher sees it and approves ----
    at.sidebar.button[-1].click().run()                          # student logs out
    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")
    at.db.set_teacher_classes("T_001", [("SE", "A")])

    at.segmented_control(key="login_role").set_value("Teacher").run()
    at.text_input[0].input("amit")
    at.text_input[1].input("pass123")
    button(at, "Login").click().run()
    at.segmented_control(key="teacher_view").set_value("🚩 Flagged attendance").run()

    assert not at.exception, at.exception
    assert any("too perfect" in w.value for w in at.warning)
    assert len(at.get("deck_gl_json_chart")) == 1                # the st.map location map

    day = record["date"]
    at.button(key=f"approve_ID_001_{day}").click().run()
    assert at.db.get_record("ID_001", day)["review"] == "approved"
    # approved = the normal rules again (Present / Pending / Absent by the clock), not Flagged
    now = rules.now_local()
    approved = at.db.get_record("ID_001", day)
    assert rules.build_report(at.db.get_people(), at.db.get_attendance(), now)["Attendance"][0]         == rules.compute_status(day, approved["time_in"], approved["time_out"], now)
    assert any("No flagged attendance" in s.value for s in at.success)


def test_reject_counts_as_absent(app4):
    at = app4
    day = rules.now_local().date().isoformat()
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    at.db.save_record("ID_001", day, "09:00:00", "09:40:00")
    at.db.save_check_info("ID_001", day, "8.8.8.8", json.dumps(
        {"saved_at": time.time(), "readings": fake_readings(*COLLEGE)}), True, "suspicious: too perfect")

    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")
    at.db.set_teacher_classes("T_001", [("SE", "A")])

    at.run()
    at.segmented_control(key="login_role").set_value("Teacher").run()
    at.text_input[0].input("amit")
    at.text_input[1].input("pass123")
    button(at, "Login").click().run()
    at.segmented_control(key="teacher_view").set_value("🚩 Flagged attendance").run()
    at.button(key=f"reject_ID_001_{day}").click().run()

    record = at.db.get_record("ID_001", day)
    assert record["review"] == "rejected" and record["flag_reason"]   # evidence kept
    assert rules.build_report(at.db.get_people(), at.db.get_attendance(),
                              rules.now_local())["Attendance"][0] == "Absent"


def test_impossible_travel_is_flagged(app4, model):
    at = app4
    img = add_student(at, model)

    # yesterday's record says: 2 minutes ago this student was at HOME (16 km away)
    yesterday = "2026-10-27"
    at.db.save_record("ID_001", yesterday, "09:00:00", "17:00:00")
    at.db.save_check_info("ID_001", yesterday, at.ip, json.dumps(
        {"saved_at": time.time() - 120, "readings": real_readings(*HOME)}), False, "")

    kind, text = login_and_mark(at, img, real_readings(*COLLEGE), real_readings(*COLLEGE))

    assert kind == "warning"
    today = rules.now_local().date().isoformat()
    assert checks.FLAG_IMPOSSIBLE_TRAVEL in at.db.get_record("ID_001", today)["flag_reason"]


def test_same_spot_flags_both_students(app4, model):
    at = app4
    img = add_student(at, model)
    today = rules.now_local().date().isoformat()

    # ID_002 marked 3 minutes ago at EXACTLY the point ID_001 will use (average of the readings)
    mine = real_readings(*COLLEGE)
    lat, lon = checks.average_position(mine)
    at.db.save_person("ID_002", "Neha P", "student", "2026-10-01", "SE", "A")
    at.db.save_record("ID_002", today, "09:00:00", None)
    at.db.save_check_info("ID_002", today, at.ip, json.dumps(
        {"saved_at": time.time() - 180,
         "readings": [{"lat": lat, "lon": lon, "accuracy": 11.0 + i, "time": time.time()}
                      for i in range(3)]}), False, "")

    kind, text = login_and_mark(at, img, real_readings(*COLLEGE), mine)

    assert kind == "warning"
    assert "same location (with ID_002)" in at.db.get_record("ID_001", today)["flag_reason"]
    other = at.db.get_record("ID_002", today)
    assert other["flagged"] == 1 and "same location (with ID_001)" in other["flag_reason"]


def test_admin_saves_college_settings_and_current_ip(app4):
    at = app4
    at.ip = "198.51.100.23"
    at.run()
    at.segmented_control(key="login_role").set_value("Teacher").run()
    at.text_input[0].input("admin")
    at.text_input[1].input("admin123")
    button(at, "Login").click().run()

    button(at, "🌐 Use my current IP").click().run()
    assert not at.exception, at.exception
    saved = at.db.get_settings()                            # saved in the database
    assert "198.51.100.23" in saved["allowed_ips"]

    # a wrong entry is refused
    v = at.session_state["college_form_v"]
    at.text_area(key=f"college_ips_{v}").input("203.0.113.0/24\nnot-an-ip")
    button(at, "💾 Save college settings").click().run()
    assert any("not-an-ip" in e.value for e in at.error)

    # Use my current location -> readings -> new centre
    button(at, "📍 Use my current location").click().run()
    send_readings(at, "admin_loc", real_readings(18.5, 73.8))
    saved = at.db.get_settings()
    assert saved["college_lat"] == pytest.approx(18.500003, abs=1e-5)


# ====================================================================== 3. no kiosk page, GPS on/off

def test_app4_has_no_mark_attendance_page(app4):
    at = app4
    at.run()
    assert not at.exception, at.exception
    assert len(at.sidebar.radio) == 0                            # no "Go to" page list
    assert "Mark Attendance" not in str([m.value for m in at.sidebar.markdown])


def test_gps_off_skips_the_location_step(app4, model):
    at = app4
    write_settings(at, gps_required=False, college_lat=None, college_lon=None)   # no centre needed
    img = add_student(at, model)

    at.run()
    at.text_input(key="login_number").input("1")
    button(at, "Continue").click().run()

    # straight to the face step: no location request at all
    assert "login_loc_state" not in at.session_state
    send_frame(at, "login_live", 1, img)
    send_frame(at, "login_live", 2, img)
    assert logged_in_user(at)["id"] == "ID_001"

    # Mark my attendance: no location step either, saved with the IP, not flagged
    assert "me_loc_state" not in at.session_state
    send_frame(at, "me_live", 1, img)
    send_frame(at, "me_live", 2, img)
    assert "Time In marked" in at.session_state["me_result"][1]

    record = at.db.get_attendance("student")[0]
    assert record["flagged"] == 0 and record["ip_address"] == "203.0.113.50"
    assert json.loads(record["gps_readings"])["readings"] == []


def test_gps_off_still_checks_college_wifi(app4):
    at = app4
    write_settings(at, gps_required=False, wifi_required=True)
    at.ip = "8.8.8.8"
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")

    at.run()
    at.text_input(key="login_number").input("1")
    button(at, "Continue").click().run()

    assert any("college Wi-Fi" in e.value for e in at.error)
    assert "login_live" not in str(at.session_state)             # no face step


def test_admin_switches_gps_off(app4):
    at = app4
    at.run()
    at.segmented_control(key="login_role").set_value("Teacher").run()
    at.text_input[0].input("admin")
    at.text_input[1].input("admin123")
    button(at, "Login").click().run()

    at.toggle(key="college_gps_0").set_value(False)
    button(at, "💾 Save college settings").click().run()

    assert not at.exception, at.exception
    assert at.db.get_settings()["gps_required"] is False


# ====================================================================== 4. sign-up approval (app4.py)

from assets.storage import SQLiteStore, sync_stores        # noqa: E402


def login_teacher(at, username, password):
    at.run()
    at.segmented_control(key="login_role").set_value("Teacher").run()
    at.text_input[0].input(username)
    at.text_input[1].input(password)
    button(at, "Login").click().run()
    assert not at.exception, at.exception


def test_sign_up_waits_for_approval_then_admin_approves(app4, model):
    at = app4
    img = one_person_photo(model)
    changed = __import__("cv2").flip(__import__("cv2").convertScaleAbs(img, alpha=0.7, beta=0), 1)

    # ---- sign up as a student ----
    at.run()
    at.segmented_control(key="login_action").set_value("Sign up").run()
    at.selectbox(key="signup_year").set_value("SE")
    at.selectbox(key="signup_div").set_value("Computer")         # a branch
    at.text_input(key="signup_name").input("Riya Sharma")
    button(at, "✅ Submit and start camera").click().run()      # the camera starts now
    send_frame(at, "reg_live", 1, img)
    for seq in (2, 3):
        at.session_state.reg_last -= 10
        send_frame(at, "reg_live", seq, changed if seq == 2 else img)
    button(at, "✅ Save sign-up").click().run()

    # saved, NOT logged in, back on "Log in" with the message
    assert not at.exception, at.exception
    assert not logged_in_user(at)
    assert at.segmented_control(key="login_action").value == "Log in"
    assert any("approves your sign-up" in s.value for s in at.success)
    assert at.db.get_person("ID_001")["approved"] == 0
    assert at.db.get_person("ID_001")["division"] == "Computer"   # the branch is saved

    # ---- login refused while pending (no location / face step) ----
    at.text_input(key="login_number").input("1")
    button(at, "Continue").click().run()
    assert any("waiting for approval" in w.value for w in at.warning)
    assert "login_loc_state" not in at.session_state

    # ---- the admin approves ----
    login_teacher(at, "admin", "admin123")
    assert any("1 sign-up(s) waiting" in w.value for w in at.warning)
    assert "🖊 Approvals (1)" in [t.label for t in at.tabs]
    at.button(key="admin_approve_ID_001").click().run()
    assert at.db.get_person("ID_001")["approved"] == 1

    # ---- now the student can start logging in (location step comes next) ----
    at.sidebar.button[-1].click().run()                         # admin logs out
    at.text_input(key="login_number").input("1")
    button(at, "Continue").click().run()
    assert at.session_state["login_pending"]["id"] == "ID_001"


def test_class_teacher_approves_students_of_own_class_only(app4):
    at = app4
    for tid, user, classes in (("T_001", "amit", [("SE", "A")]), ("T_002", "neha", [("TE", "B")])):
        at.db.save_person(tid, user.title() + " C", "teacher", "2026-10-01")
        at.db.save_teacher_account(user, auth.hash_password("pass123"), tid)
        at.db.set_teacher_classes(tid, classes)
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    at.db.set_approved("ID_001", False)

    # teacher of another class: nothing to approve
    login_teacher(at, "neha", "pass123")
    at.segmented_control(key="teacher_view").set_value("🖊 Approvals").run()
    assert any("No sign-ups waiting" in s.value for s in at.success)
    at.sidebar.button[-1].click().run()

    # teacher of SE-A approves
    login_teacher(at, "amit", "pass123")
    assert any("1 student sign-up(s)" in i.value for i in at.info)
    at.segmented_control(key="teacher_view").set_value("🖊 Approvals").run()
    at.button(key="teacher_approve_ID_001").click().run()
    assert at.db.get_person("ID_001")["approved"] == 1


def test_pending_teacher_cannot_log_in_and_reject_deletes(app4):
    at = app4
    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.set_approved("T_001", False)
    at.db.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")

    login_teacher(at, "amit", "pass123")
    assert any("waiting for approval by the admin" in w.value for w in at.warning)
    assert not logged_in_user(at)

    login_teacher(at, "admin", "admin123")
    at.checkbox(key="admin_delete_ok_T_001").check().run()          # Delete all data
    at.button(key="admin_delete_T_001").click().run()
    assert at.db.get_person("T_001") is None
    assert at.db.get_teacher_account("amit") is None


def test_pending_person_cannot_reuse_a_cookie_login(tmp_path):
    store = SQLiteStore(str(tmp_path / "p.db"))
    store.save_person("ID_001", "Riya S", "student", "2026-10-01")
    token = auth.start_login(store, {"role": "Student", "id": "ID_001", "name": "Riya S"})

    assert auth.user_from_token(store, token)["id"] == "ID_001"
    store.set_approved("ID_001", False)
    assert auth.user_from_token(store, token) is None


# ====================================================================== 5. going online

def test_old_people_stay_approved_after_upgrade(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE people (id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
                                 registered_on TEXT NOT NULL);
            INSERT INTO people VALUES ('ID_001', 'Riya S', 'student', '2026-10-01');
        """)
    assert SQLiteStore(path).get_person("ID_001")["approved"] == 1


def test_upload_keeps_approval_frozen_and_evidence(tmp_path):
    a = SQLiteStore(str(tmp_path / "a.db"))
    b = SQLiteStore(str(tmp_path / "b.db"))
    day = "2026-10-28"

    a.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")
    a.set_approved("ID_001", False)
    a.save_person("ID_002", "Neha P", "student", "2026-10-01", "SE", "A")
    a.set_frozen("ID_002", True)
    a.save_record("ID_002", day, "09:00:00", "17:00:00")
    a.save_check_info("ID_002", day, "203.0.113.9", '{"saved_at": 1, "readings": []}', True,
                      "suspicious: too perfect")
    a.set_review("ID_002", day, "rejected")

    sync_stores(a, b)

    assert b.get_person("ID_001")["approved"] == 0
    assert b.get_person("ID_002")["frozen"] == 1
    rec = b.get_record("ID_002", day)
    assert (rec["ip_address"], rec["flagged"], rec["flag_reason"], rec["review"]) == \
        ("203.0.113.9", 1, "suspicious: too perfect", "rejected")


def test_storage_mode_secret_is_used_when_nothing_is_saved(tmp_path, monkeypatch):
    from assets import storage
    monkeypatch.setattr(storage, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(storage, "online_available", lambda: True)
    monkeypatch.setattr(storage, "get_secret",
                        lambda name, default=None: "online" if name == "STORAGE_MODE" else default)

    assert storage.get_mode() == "online"            # Streamlit Cloud: from the secret
    storage.set_mode("offline")
    assert storage.get_mode() == "offline"           # the admin's choice wins



# ====================================================================== 6. branches (app4.py)

def test_branch_list_default_saved_and_secret(tmp_path, monkeypatch):
    from assets import storage

    # first use of an empty database: filled once with the default list (kept in the database)
    assert storage.get_branches() == storage.DEFAULT_BRANCHES
    assert SQLiteStore(storage.DB_PATH).get_branch_list() == storage.DEFAULT_BRANCHES

    storage.save_branches(["IT"])
    assert storage.get_branches() == ["IT"]                         # the admin's list

    storage.save_branches([])
    assert storage.get_branches() == []                             # emptied on purpose stays empty


def test_branch_list_starts_from_the_secret(monkeypatch):
    from assets import storage
    monkeypatch.setattr(storage, "get_secret",
                        lambda name, default=None: "Computer, Robotics" if name == "BRANCHES" else default)
    assert storage.get_branches() == ["Computer", "Robotics"]


def test_sync_copies_branches_and_settings(tmp_path):
    a = SQLiteStore(str(tmp_path / "a.db"))
    b = SQLiteStore(str(tmp_path / "b.db"))
    a.save_branch_list(["Computer", "IT", "Robotics"])
    a.save_app_setting("cooldown_student", 120)
    a.save_app_setting("allowed_ips", ["203.0.113.0/24"])
    b.save_branch_list(["it", "Civil"])
    b.save_app_setting("cooldown_student", 300)                     # different on purpose

    r = sync_stores(a, b)

    assert (r["branches"], r["settings"]) == (2, 1)
    assert b.get_branch_list() == ["it", "Civil", "Computer", "Robotics"]   # order kept, no duplicate
    assert b.get_settings() == {"cooldown_student": 300,                     # not overwritten
                                "allowed_ips": ["203.0.113.0/24"]}


def test_admin_adds_and_removes_branches(app4):
    at = app4
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "Civil")
    login_teacher(at, "admin", "admin123")

    at.text_input(key="branch_new").input("  Robotics  ")
    button(at, "➕ Add branch").click().run()
    assert not at.exception, at.exception

    at.text_input(key="branch_new").input("robotics")              # duplicate (any case)
    button(at, "➕ Add branch").click().run()
    assert any("already in the list" in e.value for e in at.error)

    at.button(key="branch_remove_Civil").click().run()
    assert any("still used 1×" in i.value for i in at.info)         # warned, nothing deleted

    saved = at.db.get_branch_list()                                 # saved in the database
    assert "Robotics" in saved and "Civil" not in saved
    assert at.db.get_person("ID_001")["division"] == "Civil"        # the student keeps it


def test_sign_up_offers_only_the_admins_branches(app4):
    at = app4
    at.db.save_branch_list(["Computer", "IT"])
    at.db.save_app_setting("branches_seeded", True)

    at.run()
    at.segmented_control(key="login_action").set_value("Sign up").run()
    assert at.selectbox(key="signup_div").options == ["Computer", "IT"]
    assert at.selectbox(key="signup_div").label == "Branch"

    at.segmented_control(key="signup_role").set_value("teacher").run()
    classes = at.multiselect(key="signup_classes").options
    assert "SE-Computer" in classes and len(classes) == 4 * 2       # Year x Branch

    at.db.save_branch_list([])                                       # no branches at all
    at.segmented_control(key="signup_role").set_value("student").run()
    assert any("No branches yet" in w.value for w in at.warning)


def test_old_division_values_still_work_in_people_tab(app4):
    at = app4
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "A")     # old division
    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.set_teacher_classes("T_001", [("SE", "A")])

    login_teacher(at, "admin", "admin123")
    assert not at.exception, at.exception

    branch_box = [b for b in at.selectbox if b.label == "Branch (students)"][0]
    assert "A (old)" in branch_box.options and "Computer" in branch_box.options

    # change the student to a real branch
    branch_box.set_value("Computer")
    button(at, "💾 Save changes").click().run()
    assert not at.exception, at.exception
    assert at.db.get_person("ID_001")["division"] == "Computer"


def test_database_tab_shows_branches_and_settings(app4):
    at = app4
    at.db.save_branch_list(["Computer", "Robotics"])
    at.db.save_app_setting("branches_seeded", True)
    at.db.save_app_setting("cooldown_student", 120)
    login_teacher(at, "admin", "admin123")

    at.selectbox(key="db_table").set_value("Branches").run()
    assert not at.exception, at.exception
    table = [d.value for d in at.dataframe if "branch" in d.value.columns][0]
    assert list(table["branch"]) == ["Computer", "Robotics"]

    at.selectbox(key="db_table").set_value("App settings").run()
    table = [d.value for d in at.dataframe if "setting" in d.value.columns][0]
    assert "cooldown_student" in list(table["setting"])



# ====================================================================== 7. sign-up: camera after Submit

def camera_shown(at):
    return any(c.key in ("reg_mode", "id_mode") for c in at.segmented_control) \
        or len(at.get("camera_input")) > 0


def test_signup_camera_starts_only_after_submit(app4):
    at = app4
    at.run()
    at.segmented_control(key="login_action").set_value("Sign up").run()

    # valid details typed, but not submitted -> no camera
    at.text_input(key="signup_name").input("Riya Sharma").run()
    assert not camera_shown(at)
    assert "signup_details" not in at.session_state

    # Submit -> summary + camera (a form sends its values together with the Submit click)
    at.text_input(key="signup_name").input("Riya Sharma")
    button(at, "✅ Submit and start camera").click().run()
    assert not at.exception, at.exception
    assert at.session_state["signup_details"]["name"] == "Riya Sharma"
    assert camera_shown(at)
    assert any("Riya Sharma" in i.value for i in at.info)

    # Edit details -> back to the form, camera closed
    button(at, "✏️ Edit details").click().run()
    assert "signup_details" not in at.session_state
    assert not camera_shown(at)
    assert at.text_input(key="signup_name")                     # the form is back


def test_signup_submit_checks_the_details_first(app4):
    at = app4
    at.run()
    at.segmented_control(key="login_action").set_value("Sign up").run()

    # only one name -> error, no camera
    at.text_input(key="signup_name").input("Riya")
    button(at, "✅ Submit and start camera").click().run()
    assert any("first and last name" in e.value for e in at.error)
    assert not camera_shown(at)

    # teacher with passwords that don't match -> error, no camera
    at.segmented_control(key="signup_role").set_value("teacher").run()
    at.text_input(key="signup_name").input("Neha Patil")
    at.text_input(key="signup_user").input("neha")
    at.text_input(key="signup_pass").input("secret1")
    at.text_input(key="signup_pass2").input("secret2")
    button(at, "✅ Submit and start camera").click().run()
    assert any("not the same" in e.value for e in at.error)
    assert not camera_shown(at)

    # fixed -> the teacher identity check camera starts
    at.text_input(key="signup_name").input("Neha Patil")
    at.text_input(key="signup_user").input("neha")
    at.text_input(key="signup_pass").input("secret1")
    at.text_input(key="signup_pass2").input("secret1")
    button(at, "✅ Submit and start camera").click().run()
    assert not at.exception, at.exception
    assert at.session_state["signup_details"]["username"] == "neha"
    assert camera_shown(at)



# ====================================================================== 8. reject login / delete / re-approve

def test_reject_login_says_invalid_id_and_keeps_the_data(app4):
    at = app4
    day = rules.now_local().date().isoformat()
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "Computer")
    at.db.set_approved("ID_001", False)
    at.db.save_record("ID_001", day, "09:00:00", None)
    token = auth.start_login(at.db, {"role": "Student", "id": "ID_001", "name": "Riya S"})

    # admin: ⛔ Reject login
    login_teacher(at, "admin", "admin123")
    at.button(key="admin_reject_ID_001").click().run()
    assert not at.exception, at.exception
    assert at.db.get_person("ID_001")["approved"] == -1             # still there
    assert at.db.get_record("ID_001", day)                           # data kept
    assert "🖊 Approvals (0) · ⛔ 1" in [t.label for t in at.tabs]
    assert auth.user_from_token(at.db, token) is None                # old cookie refused

    # the student's login answers exactly like an unknown ID
    at.sidebar.button[-1].click().run()                              # admin logs out
    at.text_input(key="login_number").input("1")
    button(at, "Continue").click().run()
    assert any(e.value == "Invalid Student ID" for e in at.error)
    pending = at.session_state["login_pending"] if "login_pending" in at.session_state else None
    assert not pending

    # not in the reports
    assert "ID_001" not in [p["id"] for p in at.db.get_people() if p.get("approved") == 1]


def test_rejected_teacher_gets_wrong_username_or_password(app4):
    at = app4
    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")
    at.db.reject_login("T_001")

    login_teacher(at, "amit", "pass123")
    assert any("Wrong username or password" in e.value for e in at.error)
    assert not logged_in_user(at)


def test_admin_re_approves_a_rejected_login(app4):
    at = app4
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "Computer")
    at.db.reject_login("ID_001")

    login_teacher(at, "admin", "admin123")
    at.button(key="admin_reapprove_ID_001").click().run()
    assert at.db.get_person("ID_001")["approved"] == 1

    # now the login goes on to the location / face step
    at.sidebar.button[-1].click().run()
    at.text_input(key="login_number").input("1")
    button(at, "Continue").click().run()
    assert at.session_state["login_pending"]["id"] == "ID_001"


def test_re_approve_from_people_tab_and_by_class_teacher(app4):
    at = app4
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "Computer")
    at.db.reject_login("ID_001")

    # admin -> People tab (ID_001 is the first person)
    login_teacher(at, "admin", "admin123")
    at.button(key="people_reapprove_ID_001").click().run()
    assert at.db.get_person("ID_001")["approved"] == 1

    # class teacher re-approves another rejected student of the class
    at.sidebar.button[-1].click().run()
    at.db.save_person("ID_002", "Neha P", "student", "2026-10-01", "SE", "Computer")
    at.db.reject_login("ID_002")
    at.db.save_person("T_001", "Amit C", "teacher", "2026-10-01")
    at.db.save_teacher_account("amit", auth.hash_password("pass123"), "T_001")
    at.db.set_teacher_classes("T_001", [("SE", "Computer")])

    login_teacher(at, "amit", "pass123")
    at.segmented_control(key="teacher_view").set_value("🖊 Approvals").run()
    at.button(key="teacher_reapprove_ID_002").click().run()
    assert at.db.get_person("ID_002")["approved"] == 1


def test_delete_all_data_removes_everything(app4):
    at = app4
    day = rules.now_local().date().isoformat()
    at.db.save_person("ID_001", "Riya S", "student", "2026-10-01", "SE", "Computer")
    at.db.reject_login("ID_001")
    at.db.save_record("ID_001", day, "09:00:00", None)

    login_teacher(at, "admin", "admin123")
    at.checkbox(key="admin_delete_ok_ID_001").check().run()
    at.button(key="admin_delete_ID_001").click().run()

    assert at.db.get_person("ID_001") is None
    assert at.db.get_attendance() == []


def test_sync_keeps_a_rejected_login(tmp_path):
    a = SQLiteStore(str(tmp_path / "a.db"))
    b = SQLiteStore(str(tmp_path / "b.db"))
    a.save_person("ID_001", "Riya S", "student", "2026-10-01")
    a.reject_login("ID_001")

    sync_stores(a, b)
    assert b.get_person("ID_001")["approved"] == -1
