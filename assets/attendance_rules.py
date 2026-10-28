"""
assets/attendance_rules.py - the attendance rules, kept separate from Streamlit
so they are easy to test and explain.

Rules (same as the old project):
  * first scan of the day  -> Time In
  * second scan            -> Time Out (only after Time In, and after the cooldown)
  * third scan             -> "already completed"
  * Status: Present (in + out), Pending (in, day not over), Absent
"""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

# -------- SETTINGS --------
COOLDOWN_SECONDS = 60          # default wait between Time In and Time Out
                               # (the admin can set one for students and one for teachers)
CUTOFF_TIME_IN = time(15, 30)  # no Time In by 15:30 -> Absent
CUTOFF_TIME_OUT = time(17, 30) # Time In but no Time Out by 17:30 -> Absent
MAX_HOURS = 12                 # Time In older than 12 h without Time Out -> Absent
TIMEZONE = "Asia/Kolkata"      # cloud servers run in UTC, so we fix the timezone

REPORT_COLUMNS = ["ID", "Name", "Role", "Date", "Time In", "Time Out", "Attendance"]


def now_local():
    """Current time in TIMEZONE (without tz info, to keep comparisons simple)."""

    return datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)


def _parse_time(text):
    return datetime.strptime(text, "%H:%M:%S").time()


def wait_text(seconds):
    """45 -> '45 seconds', 200 -> '3 min 20 s', 3600 -> '60 min'"""

    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} seconds"

    minutes, rest = divmod(seconds, 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


def decide_scan(record, now, cooldown=COOLDOWN_SECONDS):
    """
    What should happen when this person scans now?
    record   = today's attendance row for the person (dict) or None.
    cooldown = seconds to wait between Time In and Time Out (different for students / teachers).
    Returns (action, seconds_left):
      "time_in"  - mark Time In
      "time_out" - mark Time Out
      "cooldown" - too soon after Time In, seconds_left tells how long to wait
      "done"     - Time In and Time Out already marked today
    """

    if record is None or not record.get("time_in"):
        return "time_in", 0

    if record.get("time_out"):
        return "done", 0

    time_in = datetime.combine(now.date(), _parse_time(record["time_in"]))
    elapsed = (now - time_in).total_seconds()

    if elapsed < cooldown:
        return "cooldown", int(cooldown - elapsed)

    return "time_out", 0


def compute_status(day, time_in, time_out, now):
    """Present / Pending / Absent for one person on one day ("YYYY-MM-DD")."""

    if time_in and time_out:
        return "Present"

    day = date.fromisoformat(str(day))

    if time_in:
        in_time = datetime.combine(day, _parse_time(time_in))
        hours_passed = (now - in_time).total_seconds() / 3600

        if hours_passed >= MAX_HOURS or now >= datetime.combine(day, CUTOFF_TIME_OUT):
            return "Absent"

        return "Pending"

    if now >= datetime.combine(day, CUTOFF_TIME_IN):
        return "Absent"

    return "Pending"


def apply_review(status, record):
    """
    app4.py anti-fake-GPS checks:
      * a record a teacher REJECTED counts as Absent
      * a FLAGGED record that nobody reviewed yet shows as "Flagged"
      * an APPROVED one keeps its normal status (Present / Pending)
    Records without these fields (app.py / app2.py) are not changed.
    """

    if record.get("review") == "rejected":
        return "Absent"
    if record.get("flagged") and record.get("review") != "approved":
        return "Flagged"
    return status


def build_report(people, records, now):
    """
    One row per person per working day.
    A "working day" is any day on which at least one person scanned.
    People with no scan that day are listed too (usually Absent),
    but only for days on or after they registered.

    people  : list of dicts {id, name, role, registered_on}
    records : list of dicts {person_id, date, time_in, time_out}
    """

    by_key = {(r["person_id"], str(r["date"])): r for r in records}
    days = sorted({str(r["date"]) for r in records})

    rows = []

    for day in days:
        for p in people:

            if str(p["registered_on"]) > day:
                continue

            r = by_key.get((p["id"], day), {})
            time_in = r.get("time_in") or ""
            time_out = r.get("time_out") or ""

            rows.append({
                "ID": p["id"],
                "Name": p["name"],
                "Role": p["role"],
                "Date": day,
                "Time In": time_in,
                "Time Out": time_out,
                "Attendance": apply_review(compute_status(day, time_in, time_out, now), r)
            })

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def count_status(df, with_flagged=False):
    """{"Present": n, "Pending": n, "Absent": n} for a report DataFrame
    (+ "Flagged": n when with_flagged, used by app4.py)."""

    statuses = ["Present", "Pending", "Absent"] + (["Flagged"] if with_flagged else [])
    return {s: int((df["Attendance"] == s).sum()) for s in statuses}
