"""
assets/storage.py - where people, face embeddings and attendance are saved.

Two "backends" with exactly the same functions:
  * SQLiteStore   - a local file data.db (default, for testing on your laptop)
  * SupabaseStore - a free online Postgres database (for Streamlit Cloud,
                    because the cloud deletes local files when the app restarts)

The admin chooses which one the whole app uses (Admin -> Database tab).
The choice is saved in settings.json; without it the app uses SQLite.
sync_stores() copies missing data from one to the other (it never deletes).

Tables (same in both):
  people             (id, name, role, registered_on, year, division, frozen, approved)
                       approved: 1 = may log in, 0 = sign-up waiting, -1 = login rejected (app4.py)
  embeddings         (id, person_id, embedding)            - several rows per person
  student_attendance (person_id, date, time_in, time_out)  - one row per student per day
  teacher_attendance (person_id, date, time_in, time_out)  - one row per teacher per day
  teacher_classes    (teacher_id, year, division)          - classes a teacher teaches
  teacher_accounts   (username, password_hash, person_id)  - teacher logins (username + password)
  login_sessions     (token_hash, role, person_id, expires_at) - "stay logged in" (app2.py)
  branches           (name, position)                      - app4.py branch list (admin)
  app_settings       (key, value)                          - admin's App settings (JSON values)
  attendance_backup  - the old single attendance table, kept after the upgrade (not used)

The attendance functions take a person ID and pick the right table from the person's role,
so the rest of the app does not need to know there are two tables.
"""

import base64
import json
import os
import sqlite3
import time

import numpy as np
import streamlit as st

from assets.attendance_rules import COOLDOWN_SECONDS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder (one above assets/)
DB_PATH = os.environ.get("ATTENDANCE_DB", os.path.join(BASE_DIR, "data.db"))
SETTINGS_PATH = os.environ.get("ATTENDANCE_SETTINGS", os.path.join(BASE_DIR, "settings.json"))

# extra columns on student_attendance / teacher_attendance (app4.py anti-fake-GPS checks)
CHECK_COLUMNS = [
    ("ip_address", "TEXT"),                      # internet address used to mark
    ("gps_readings", "TEXT"),                    # the 3 GPS readings as JSON text
    ("flagged", "INTEGER NOT NULL DEFAULT 0"),   # 1 = suspicious, a teacher must review it
    ("flag_reason", "TEXT"),                     # why it was flagged
    ("review", "TEXT"),                          # '' / 'approved' / 'rejected'
]
ATTENDANCE_COLUMNS = ", ".join(["person_id", "date", "time_in", "time_out"]
                               + [c for c, _ in CHECK_COLUMNS])

YEARS = ["FE", "SE", "TE", "BE"]

# app4.py uses BRANCHES instead of divisions A-D. The branch name is stored in the same
# "division" column (people + teacher_classes), so the database didn't change.
# The admin edits the list in App settings (saved in settings.json).
DEFAULT_BRANCHES = ["Computer", "IT", "AI & DS", "Electronics", "Mechanical", "Civil", "Electrical"]
DIVISIONS = ["A", "B", "C", "D"]


def get_secret(name, default=None):
    """Read st.secrets without crashing when there is no secrets.toml."""

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def _from_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def make_next_id(existing_ids, role):
    """Students: ID_001, ID_002 ...  Teachers: T_001, T_002 ..."""

    prefix = "ID_" if role == "student" else "T_"
    numbers = [int(i[len(prefix):]) for i in existing_ids
               if i.startswith(prefix) and i[len(prefix):].isdigit()]

    return f"{prefix}{max(numbers, default=0) + 1:03}"


def attendance_table(role):
    return "teacher_attendance" if role == "teacher" else "student_attendance"


def class_name(year, division):
    """('SE', 'A') -> 'SE-A'"""
    return f"{year}-{division}"


def students_of_teacher(store, teacher_id):
    """Students whose year + division is one of the teacher's classes."""

    classes = {(c["year"], c["division"]) for c in store.get_teacher_classes(teacher_id)}

    return [p for p in store.get_people()
            if p["role"] == "student" and (p.get("year"), p.get("division")) in classes]


# ======================================================================
# Backend A: SQLite (local file)
# ======================================================================

class SQLiteStore:

    name = "SQLite (local file data.db)"

    def __init__(self, path=DB_PATH):
        self.path = path
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    registered_on TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL REFERENCES people(id),
                    embedding BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS student_attendance (
                    person_id TEXT NOT NULL REFERENCES people(id),
                    date TEXT NOT NULL,
                    time_in TEXT,
                    time_out TEXT,
                    PRIMARY KEY (person_id, date)
                );
                CREATE TABLE IF NOT EXISTS teacher_attendance (
                    person_id TEXT NOT NULL REFERENCES people(id),
                    date TEXT NOT NULL,
                    time_in TEXT,
                    time_out TEXT,
                    PRIMARY KEY (person_id, date)
                );
                CREATE TABLE IF NOT EXISTS teacher_classes (
                    teacher_id TEXT NOT NULL REFERENCES people(id),
                    year TEXT NOT NULL,
                    division TEXT NOT NULL,
                    PRIMARY KEY (teacher_id, year, division)
                );
                CREATE TABLE IF NOT EXISTS teacher_accounts (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    person_id TEXT NOT NULL REFERENCES people(id)
                );
                CREATE TABLE IF NOT EXISTS login_sessions (
                    token_hash TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open'
                );
                CREATE TABLE IF NOT EXISTS branches (
                    name TEXT PRIMARY KEY,
                    position INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            self._upgrade(db)
            self._one_account_per_teacher(db)

    def _one_account_per_teacher(self, db):
        """A teacher ID may have only ONE login account (the database refuses a second one).
        The rule is only added when there are no duplicates yet - existing rows are never deleted."""

        duplicates = db.execute("SELECT person_id FROM teacher_accounts "
                                "GROUP BY person_id HAVING COUNT(*) > 1").fetchone()
        if not duplicates:
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS teacher_accounts_one_per_teacher "
                       "ON teacher_accounts(person_id)")

    def _upgrade(self, db):
        """
        One-time upgrade of an older data.db (nothing is deleted):
          * people gets the year, division, frozen and approved columns
          * rows of the old 'attendance' table are copied into student_attendance /
            teacher_attendance, then the old table is renamed to attendance_backup
        """

        columns = [r["name"] for r in db.execute("PRAGMA table_info(people)")]
        for column in ("year", "division"):
            if column not in columns:
                db.execute(f"ALTER TABLE people ADD COLUMN {column} TEXT")
        if "frozen" not in columns:
            db.execute("ALTER TABLE people ADD COLUMN frozen INTEGER NOT NULL DEFAULT 0")
        if "approved" not in columns:
            # app4.py: a new sign-up must be approved before logging in (old people: approved)
            db.execute("ALTER TABLE people ADD COLUMN approved INTEGER NOT NULL DEFAULT 1")

        # anti-fake-GPS evidence on every attendance row (app4.py), same order in both tables
        for table in ("student_attendance", "teacher_attendance"):
            have = [r["name"] for r in db.execute(f"PRAGMA table_info({table})")]
            for column, kind in CHECK_COLUMNS:
                if column not in have:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

        old = db.execute("SELECT name FROM sqlite_master "
                         "WHERE type = 'table' AND name = 'attendance'").fetchone()
        if not old:
            return

        # safety copy of the whole file before moving anything
        backup = os.path.splitext(self.path)[0] + "_backup_before_upgrade.db"
        if not os.path.exists(backup):
            with sqlite3.connect(backup) as copy:
                db.backup(copy)

        for role in ("student", "teacher"):
            db.execute(f"""
                INSERT OR IGNORE INTO {attendance_table(role)} (person_id, date, time_in, time_out)
                SELECT a.person_id, a.date, a.time_in, a.time_out
                FROM attendance a JOIN people p ON p.id = a.person_id
                WHERE p.role = ?""", (role,))

        db.execute("ALTER TABLE attendance RENAME TO attendance_backup")

    def _connect(self):
        # a new connection per call is simple and safe with Streamlit's threads
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def _table_of(self, person_id):
        person = self.get_person(person_id)
        return attendance_table(person["role"] if person else "student")

    # ---- people ----
    def next_person_id(self, role):
        with self._connect() as db:
            ids = [r["id"] for r in db.execute("SELECT id FROM people")]
        return make_next_id(ids, role)

    def save_person(self, person_id, name, role, registered_on, year=None, division=None):
        with self._connect() as db:
            db.execute("INSERT INTO people (id, name, role, registered_on, year, division) "
                       "VALUES (?, ?, ?, ?, ?, ?)",
                       (person_id, name, role, registered_on, year, division))

    def get_people(self):
        with self._connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM people ORDER BY id")]

    def get_person(self, person_id):
        with self._connect() as db:
            # IDs are always upper case (ID_001, T_001)
            r = db.execute("SELECT * FROM people WHERE id = ?",
                           (person_id.strip().upper(),)).fetchone()
        return dict(r) if r else None

    # ---- embeddings (stored as raw float32 bytes = BLOB) ----
    def save_embeddings(self, person_id, embeddings):
        with self._connect() as db:
            db.executemany(
                "INSERT INTO embeddings (person_id, embedding) VALUES (?, ?)",
                [(person_id, np.asarray(e, dtype=np.float32).tobytes()) for e in embeddings]
            )

    def get_all_embeddings(self):
        with self._connect() as db:
            rows = db.execute("SELECT person_id, embedding FROM embeddings").fetchall()
        ids = np.array([r["person_id"] for r in rows], dtype=str)
        embs = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows],
                        dtype=np.float32).reshape(-1, 512)
        return ids, embs

    # ---- attendance (table picked from the person's role) ----
    def get_record(self, person_id, day):
        with self._connect() as db:
            r = db.execute(f"SELECT * FROM {self._table_of(person_id)} "
                           "WHERE person_id = ? AND date = ?", (person_id, day)).fetchone()
        return dict(r) if r else None

    def mark_time_in(self, person_id, day, time_text):
        with self._connect() as db:
            db.execute(f"INSERT INTO {self._table_of(person_id)} (person_id, date, time_in) "
                       "VALUES (?, ?, ?)", (person_id, day, time_text))

    def mark_time_out(self, person_id, day, time_text):
        with self._connect() as db:
            db.execute(f"UPDATE {self._table_of(person_id)} SET time_out = ? "
                       "WHERE person_id = ? AND date = ?", (time_text, person_id, day))

    def get_attendance(self, role=None):
        """Records of one role, or of both tables together when role is None."""
        roles = [role] if role else ["student", "teacher"]
        with self._connect() as db:
            return [dict(r) for role in roles
                    for r in db.execute(f"SELECT * FROM {attendance_table(role)}")]

    # ---- admin: edit / delete ----
    def update_person(self, person_id, name, role):
        old_table = self._table_of(person_id)
        with self._connect() as db:
            db.execute("UPDATE people SET name = ?, role = ? WHERE id = ?",
                       (name, role, person_id))

            # role changed -> move the person's records to the other attendance table
            new_table = attendance_table(role)
            if new_table != old_table:
                columns = ATTENDANCE_COLUMNS
                db.execute(f"INSERT OR REPLACE INTO {new_table} ({columns}) "
                           f"SELECT {columns} FROM {old_table} WHERE person_id = ?", (person_id,))
                db.execute(f"DELETE FROM {old_table} WHERE person_id = ?", (person_id,))

    def set_class(self, person_id, year, division):
        """Year + division of a student."""
        with self._connect() as db:
            db.execute("UPDATE people SET year = ?, division = ? WHERE id = ?",
                       (year, division, person_id))

    def delete_person(self, person_id):
        # delete everything that belongs to the person first
        with self._connect() as db:
            db.execute("DELETE FROM login_sessions WHERE person_id = ?", (person_id,))
            db.execute("DELETE FROM teacher_accounts WHERE person_id = ?", (person_id,))
            db.execute("DELETE FROM teacher_classes WHERE teacher_id = ?", (person_id,))
            db.execute("DELETE FROM student_attendance WHERE person_id = ?", (person_id,))
            db.execute("DELETE FROM teacher_attendance WHERE person_id = ?", (person_id,))
            db.execute("DELETE FROM embeddings WHERE person_id = ?", (person_id,))
            db.execute("DELETE FROM people WHERE id = ?", (person_id,))

    def save_record(self, person_id, day, time_in, time_out):
        """Add a record, or replace it if the person already has one that day."""
        with self._connect() as db:
            db.execute(f"INSERT OR REPLACE INTO {self._table_of(person_id)} "
                       "(person_id, date, time_in, time_out) VALUES (?, ?, ?, ?)",
                       (person_id, day, time_in or None, time_out or None))

    def delete_record(self, person_id, day):
        with self._connect() as db:
            db.execute(f"DELETE FROM {self._table_of(person_id)} WHERE person_id = ? AND date = ?",
                       (person_id, day))

    # ---- anti-fake-GPS evidence (app4.py) ----
    def save_check_info(self, person_id, day, ip, gps_json, flagged, reason):
        """Store the IP, GPS readings and flag of today's record (after Time In / Time Out)."""
        with self._connect() as db:
            db.execute(f"UPDATE {self._table_of(person_id)} SET ip_address = ?, gps_readings = ?, "
                       "flagged = ?, flag_reason = ?, review = '' WHERE person_id = ? AND date = ?",
                       (ip, gps_json, 1 if flagged else 0, reason or None, person_id, day))

    def set_review(self, person_id, day, decision):
        """A teacher's decision on a flagged record: 'approved' or 'rejected'."""
        with self._connect() as db:
            db.execute(f"UPDATE {self._table_of(person_id)} SET review = ? "
                       "WHERE person_id = ? AND date = ?", (decision, person_id, day))

    # ---- teacher classes ----
    def get_teacher_classes(self, teacher_id=None):
        with self._connect() as db:
            if teacher_id:
                rows = db.execute("SELECT * FROM teacher_classes WHERE teacher_id = ?",
                                  (teacher_id,))
            else:
                rows = db.execute("SELECT * FROM teacher_classes")
            return [dict(r) for r in rows]

    def set_teacher_classes(self, teacher_id, classes):
        """classes: list of (year, division). Replaces the teacher's list."""
        with self._connect() as db:
            db.execute("DELETE FROM teacher_classes WHERE teacher_id = ?", (teacher_id,))
            db.executemany("INSERT INTO teacher_classes VALUES (?, ?, ?)",
                           [(teacher_id, y, d) for y, d in classes])

    # ---- teacher accounts ----
    def get_teacher_accounts(self):
        with self._connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM teacher_accounts ORDER BY username")]

    def get_teacher_account(self, username):
        with self._connect() as db:
            r = db.execute("SELECT * FROM teacher_accounts WHERE username = ?",
                           (username.strip().lower(),)).fetchone()
        return dict(r) if r else None

    def get_account_of_teacher(self, person_id):
        """The login account of a teacher ID, or None."""
        with self._connect() as db:
            r = db.execute("SELECT * FROM teacher_accounts WHERE person_id = ?",
                           (person_id,)).fetchone()
        return dict(r) if r else None

    def save_teacher_account(self, username, password_hash, person_id):
        """Create an account, or change the password if the username exists.
        (Not INSERT OR REPLACE: that would silently delete a teacher's other account.)"""
        with self._connect() as db:
            db.execute("INSERT INTO teacher_accounts VALUES (?, ?, ?) "
                       "ON CONFLICT(username) DO UPDATE SET "
                       "password_hash = excluded.password_hash, person_id = excluded.person_id",
                       (username.strip().lower(), password_hash, person_id))

    def delete_teacher_account(self, username):
        with self._connect() as db:
            db.execute("DELETE FROM teacher_accounts WHERE username = ?", (username,))

    # ---- login sessions ("stay logged in"; only a hash of the token is stored) ----
    def save_login_session(self, token_hash, role, person_id, expires_at):
        with self._connect() as db:
            db.execute("INSERT INTO login_sessions VALUES (?, ?, ?, ?)",
                       (token_hash, role, person_id, expires_at))

    def get_login_session(self, token_hash):
        with self._connect() as db:
            r = db.execute("SELECT * FROM login_sessions WHERE token_hash = ?",
                           (token_hash,)).fetchone()
        return dict(r) if r else None

    def delete_login_session(self, token_hash):
        with self._connect() as db:
            db.execute("DELETE FROM login_sessions WHERE token_hash = ?", (token_hash,))

    def delete_login_sessions_of(self, person_id):
        """Log a person out on every browser."""
        with self._connect() as db:
            db.execute("DELETE FROM login_sessions WHERE person_id = ?", (person_id,))

    # ---- frozen accounts + alerts for the admin ----
    def set_frozen(self, person_id, frozen):
        with self._connect() as db:
            db.execute("UPDATE people SET frozen = ? WHERE id = ?", (1 if frozen else 0, person_id))

    def set_approved(self, person_id, approved):
        """app4.py: 0 = new sign-up waiting for approval, 1 = may log in."""
        with self._connect() as db:
            db.execute("UPDATE people SET approved = ? WHERE id = ?",
                       (1 if approved else 0, person_id))

    def reject_login(self, person_id):
        """app4.py: -1 = login rejected (all data kept; the login answers "Invalid ID")."""
        with self._connect() as db:
            db.execute("UPDATE people SET approved = -1 WHERE id = ?", (person_id,))

    def add_alert(self, student_id, message, created_at):
        with self._connect() as db:
            db.execute("INSERT INTO alerts (created_at, student_id, message, status) "
                       "VALUES (?, ?, ?, 'open')", (created_at, student_id, message))

    def get_alerts(self, status=None):
        """Newest first. status: 'open', 'restored', 'deleted' or None for all."""
        with self._connect() as db:
            if status:
                rows = db.execute("SELECT * FROM alerts WHERE status = ? ORDER BY id DESC", (status,))
            else:
                rows = db.execute("SELECT * FROM alerts ORDER BY id DESC")
            return [dict(r) for r in rows]

    def set_alert_status(self, alert_id, status):
        with self._connect() as db:
            db.execute("UPDATE alerts SET status = ? WHERE id = ?", (status, alert_id))

    # ---- branches + App settings (kept in the database so they go online too) ----
    def get_branch_list(self):
        with self._connect() as db:
            return [r["name"] for r in db.execute("SELECT name FROM branches ORDER BY position, name")]

    def save_branch_list(self, names):
        """Replace the whole list (keeps the order)."""
        with self._connect() as db:
            db.execute("DELETE FROM branches")
            db.executemany("INSERT INTO branches (name, position) VALUES (?, ?)",
                           [(n, i) for i, n in enumerate(names)])

    def get_settings(self):
        """{key: value} - values are stored as JSON text."""
        with self._connect() as db:
            rows = db.execute("SELECT key, value FROM app_settings").fetchall()
        return {r["key"]: _from_json(r["value"]) for r in rows}

    def save_app_setting(self, key, value):
        with self._connect() as db:
            db.execute("INSERT INTO app_settings (key, value) VALUES (?, ?) "
                       "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                       (key, json.dumps(value)))


# ======================================================================
# Backend B: Supabase (online, survives app restarts)
# The tables are created with the SQL in database.md
# ======================================================================

class SupabaseStore:

    name = "Supabase (online database)"

    def __init__(self, url, key):
        from supabase import create_client
        self.client = create_client(url, key)

    def _select_all(self, table, columns="*"):
        # Supabase returns max 1000 rows per request, so read page by page
        rows, start, page = [], 0, 1000
        while True:
            batch = (self.client.table(table).select(columns)
                     .range(start, start + page - 1).execute().data)
            rows += batch
            if len(batch) < page:
                return rows
            start += page

    def _table_of(self, person_id):
        person = self.get_person(person_id)
        return attendance_table(person["role"] if person else "student")

    # ---- people ----
    def next_person_id(self, role):
        return make_next_id([r["id"] for r in self._select_all("people", "id")], role)

    def save_person(self, person_id, name, role, registered_on, year=None, division=None):
        self.client.table("people").insert({
            "id": person_id, "name": name, "role": role, "registered_on": registered_on,
            "year": year, "division": division
        }).execute()

    def get_people(self):
        return sorted(self._select_all("people"), key=lambda r: r["id"])

    def get_person(self, person_id):
        # IDs are always upper case (ID_001, T_001)
        rows = (self.client.table("people").select("*")
                .eq("id", person_id.strip().upper()).execute().data)
        return rows[0] if rows else None

    # ---- embeddings (stored as base64 text, because the REST API sends JSON) ----
    def save_embeddings(self, person_id, embeddings):
        self.client.table("embeddings").insert([
            {"person_id": person_id,
             "embedding": base64.b64encode(np.asarray(e, dtype=np.float32).tobytes()).decode()}
            for e in embeddings
        ]).execute()

    def get_all_embeddings(self):
        rows = self._select_all("embeddings", "person_id, embedding")
        ids = np.array([r["person_id"] for r in rows], dtype=str)
        embs = np.array([np.frombuffer(base64.b64decode(r["embedding"]), dtype=np.float32)
                         for r in rows], dtype=np.float32).reshape(-1, 512)
        return ids, embs

    # ---- attendance (table picked from the person's role) ----
    def get_record(self, person_id, day):
        rows = (self.client.table(self._table_of(person_id)).select("*")
                .eq("person_id", person_id).eq("date", day).execute().data)
        return rows[0] if rows else None

    def mark_time_in(self, person_id, day, time_text):
        self.client.table(self._table_of(person_id)).insert({
            "person_id": person_id, "date": day, "time_in": time_text
        }).execute()

    def mark_time_out(self, person_id, day, time_text):
        (self.client.table(self._table_of(person_id)).update({"time_out": time_text})
         .eq("person_id", person_id).eq("date", day).execute())

    def get_attendance(self, role=None):
        """Records of one role, or of both tables together when role is None."""
        roles = [role] if role else ["student", "teacher"]
        return [r for role in roles for r in self._select_all(attendance_table(role))]

    # ---- admin: edit / delete ----
    def update_person(self, person_id, name, role):
        old_table = self._table_of(person_id)
        (self.client.table("people").update({"name": name, "role": role})
         .eq("id", person_id).execute())

        # role changed -> move the person's records to the other attendance table
        new_table = attendance_table(role)
        if new_table != old_table:
            rows = (self.client.table(old_table).select("*")
                    .eq("person_id", person_id).execute().data)
            if rows:
                self.client.table(new_table).upsert(rows).execute()
                self.client.table(old_table).delete().eq("person_id", person_id).execute()

    def set_class(self, person_id, year, division):
        """Year + division of a student."""
        (self.client.table("people").update({"year": year, "division": division})
         .eq("id", person_id).execute())

    def delete_person(self, person_id):
        # "on delete cascade" in the SQL also removes embeddings, attendance, classes and account
        self.client.table("login_sessions").delete().eq("person_id", person_id).execute()
        self.client.table("people").delete().eq("id", person_id).execute()

    def save_record(self, person_id, day, time_in, time_out):
        """Add a record, or replace it if the person already has one that day."""
        self.client.table(self._table_of(person_id)).upsert({
            "person_id": person_id, "date": day,
            "time_in": time_in or None, "time_out": time_out or None
        }).execute()

    def delete_record(self, person_id, day):
        (self.client.table(self._table_of(person_id)).delete()
         .eq("person_id", person_id).eq("date", day).execute())

    # ---- anti-fake-GPS evidence (app4.py) ----
    def save_check_info(self, person_id, day, ip, gps_json, flagged, reason):
        """Store the IP, GPS readings and flag of today's record (after Time In / Time Out)."""
        (self.client.table(self._table_of(person_id)).update({
            "ip_address": ip, "gps_readings": gps_json, "flagged": 1 if flagged else 0,
            "flag_reason": reason or None, "review": ""
        }).eq("person_id", person_id).eq("date", day).execute())

    def set_review(self, person_id, day, decision):
        """A teacher's decision on a flagged record: 'approved' or 'rejected'."""
        (self.client.table(self._table_of(person_id)).update({"review": decision})
         .eq("person_id", person_id).eq("date", day).execute())

    # ---- teacher classes ----
    def get_teacher_classes(self, teacher_id=None):
        if teacher_id:
            return (self.client.table("teacher_classes").select("*")
                    .eq("teacher_id", teacher_id).execute().data)
        return self._select_all("teacher_classes")

    def set_teacher_classes(self, teacher_id, classes):
        """classes: list of (year, division). Replaces the teacher's list."""
        self.client.table("teacher_classes").delete().eq("teacher_id", teacher_id).execute()
        if classes:
            self.client.table("teacher_classes").insert(
                [{"teacher_id": teacher_id, "year": y, "division": d} for y, d in classes]
            ).execute()

    # ---- teacher accounts ----
    def get_teacher_accounts(self):
        return sorted(self._select_all("teacher_accounts"), key=lambda r: r["username"])

    def get_teacher_account(self, username):
        rows = (self.client.table("teacher_accounts").select("*")
                .eq("username", username.strip().lower()).execute().data)
        return rows[0] if rows else None

    def get_account_of_teacher(self, person_id):
        """The login account of a teacher ID, or None."""
        rows = (self.client.table("teacher_accounts").select("*")
                .eq("person_id", person_id).execute().data)
        return rows[0] if rows else None

    def save_teacher_account(self, username, password_hash, person_id):
        """Create an account, or change its password / teacher if it exists."""
        self.client.table("teacher_accounts").upsert({
            "username": username.strip().lower(),
            "password_hash": password_hash,
            "person_id": person_id
        }).execute()

    def delete_teacher_account(self, username):
        self.client.table("teacher_accounts").delete().eq("username", username).execute()

    # ---- login sessions ("stay logged in"; only a hash of the token is stored) ----
    def save_login_session(self, token_hash, role, person_id, expires_at):
        self.client.table("login_sessions").insert({
            "token_hash": token_hash, "role": role,
            "person_id": person_id, "expires_at": expires_at
        }).execute()

    def get_login_session(self, token_hash):
        rows = (self.client.table("login_sessions").select("*")
                .eq("token_hash", token_hash).execute().data)
        return rows[0] if rows else None

    def delete_login_session(self, token_hash):
        self.client.table("login_sessions").delete().eq("token_hash", token_hash).execute()

    def delete_login_sessions_of(self, person_id):
        """Log a person out on every browser."""
        self.client.table("login_sessions").delete().eq("person_id", person_id).execute()

    # ---- frozen accounts + alerts for the admin ----
    def set_frozen(self, person_id, frozen):
        (self.client.table("people").update({"frozen": 1 if frozen else 0})
         .eq("id", person_id).execute())

    def set_approved(self, person_id, approved):
        """app4.py: 0 = new sign-up waiting for approval, 1 = may log in."""
        (self.client.table("people").update({"approved": 1 if approved else 0})
         .eq("id", person_id).execute())

    def reject_login(self, person_id):
        """app4.py: -1 = login rejected (all data kept; the login answers "Invalid ID")."""
        self.client.table("people").update({"approved": -1}).eq("id", person_id).execute()

    def add_alert(self, student_id, message, created_at):
        self.client.table("alerts").insert({
            "created_at": created_at, "student_id": student_id,
            "message": message, "status": "open"
        }).execute()

    def get_alerts(self, status=None):
        """Newest first. status: 'open', 'restored', 'deleted' or None for all."""
        query = self.client.table("alerts").select("*")
        if status:
            query = query.eq("status", status)
        return query.order("id", desc=True).execute().data

    def set_alert_status(self, alert_id, status):
        self.client.table("alerts").update({"status": status}).eq("id", alert_id).execute()

    # ---- branches + App settings (kept in the database so they go online too) ----
    def get_branch_list(self):
        rows = self.client.table("branches").select("name, position").execute().data
        return [r["name"] for r in sorted(rows, key=lambda r: (r["position"], r["name"]))]

    def save_branch_list(self, names):
        """Replace the whole list (keeps the order)."""
        self.client.table("branches").delete().neq("name", "").execute()
        if names:
            self.client.table("branches").insert(
                [{"name": n, "position": i} for i, n in enumerate(names)]).execute()

    def get_settings(self):
        """{key: value} - values are stored as JSON text."""
        rows = self._select_all("app_settings")
        return {r["key"]: _from_json(r["value"]) for r in rows}

    def save_app_setting(self, key, value):
        self.client.table("app_settings").upsert(
            {"key": key, "value": json.dumps(value)}).execute()


# ======================================================================
# Choosing the database (offline = SQLite, online = Supabase)
# ======================================================================

def supabase_config():
    """(url, key) from secrets - at the top level or inside a [supabase] section."""

    section = get_secret("supabase", {}) or {}
    url = get_secret("SUPABASE_URL") or section.get("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY") or section.get("SUPABASE_KEY")

    return url, key


def online_available():
    url, key = supabase_config()
    return bool(url and key)


# ======================================================================
# App settings chosen by the admin.
#
# They live IN THE DATABASE (table app_settings), so they are the same for everybody
# and go online with Upload / Sync:
#   capture_mode, capture_switch, cooldown_student, cooldown_teacher,
#   gps_required, college_lat, college_lon, geofence_radius_m, allowed_ips,
#   wifi_required, branches_seeded            (+ the branch list in table branches)
#
# Only "storage" (offline / online) stays in settings.json: it decides WHICH
# database to open, so it can't be stored inside one.
# Older values in settings.json are still read until they are saved again.
# ======================================================================

SETTINGS_CACHE_SECONDS = float(os.environ.get("ATTENDANCE_SETTINGS_CACHE", 5))
_settings_stores = {}      # one store object per database (path or URL)
_settings_cache = {}       # (database, what) -> (time read, value)


def load_settings():
    """settings.json (the storage mode + older settings)."""
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            settings = json.load(f)
        return settings if isinstance(settings, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_file_setting(name, value):
    settings = load_settings()
    settings[name] = value
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def settings_store():
    """(key, store) of the database the app uses right now (it also holds the settings)."""

    mode = get_mode()
    key = (mode, supabase_config()[0] if mode == "online" else DB_PATH)

    if key not in _settings_stores:
        _settings_stores[key] = (SupabaseStore(*supabase_config()) if mode == "online"
                                 else SQLiteStore(DB_PATH))
        _copy_old_settings(_settings_stores[key])
    return key, _settings_stores[key]


def _copy_old_settings(store):
    """
    Once per database: settings saved in settings.json BEFORE they moved to the database
    are copied in (only keys the database doesn't have yet - nothing is overwritten),
    so Upload takes them online too.
    """

    old = {k: v for k, v in load_settings().items() if k not in ("storage", "branches")}
    if not old:
        return

    try:
        have = store.get_settings()
        for name, value in old.items():
            if name not in have:
                store.save_app_setting(name, value)
    except Exception:
        pass                   # database not reachable now: settings.json is still read meanwhile


def _cached(what, read):
    """Read from the database at most every few seconds (pages ask for settings often)."""

    try:
        key, store = settings_store()
    except Exception:
        return None
    cached = _settings_cache.get((key, what))

    if cached and time.time() - cached[0] < SETTINGS_CACHE_SECONDS:
        return cached[1]

    try:
        value = read(store)
    except Exception:          # database not reachable: fall back to settings.json / defaults
        return None

    _settings_cache[(key, what)] = (time.time(), value)
    return value


def _forget_cache():
    _settings_cache.clear()


def get_setting(name, default):
    if name == "storage":
        return load_settings().get(name, default)

    values = _cached("settings", lambda store: store.get_settings()) or {}
    if name in values:
        return values[name]

    return load_settings().get(name, default)         # older value from settings.json


def save_setting(name, value):
    """Change ONE setting (in the database; 'storage' in settings.json)."""

    if name == "storage":
        _save_file_setting(name, value)
        _forget_cache()                               # another database may be used now
        return

    _, store = settings_store()
    store.save_app_setting(name, value)
    _forget_cache()


def get_mode():
    """
    'offline' or 'online':
      1. what the admin saved (settings.json)
      2. otherwise the secret STORAGE_MODE (Streamlit Cloud has no settings.json,
         so STORAGE_MODE = "online" makes the cloud app start online by itself)
      3. otherwise offline
    """

    mode = get_setting("storage", None)
    if mode is None:
        mode = str(get_secret("STORAGE_MODE", "offline")).strip().lower()

    # online chosen but no Supabase secrets -> use the local file
    if mode == "online" and not online_available():
        return "offline"

    return mode if mode in ("offline", "online") else "offline"


def set_mode(mode):
    save_setting("storage", mode)


def get_capture_default():
    """'Auto' unless the admin chose 'Manual'."""
    return "Manual" if get_setting("capture_mode", "Auto") == "Manual" else "Auto"


def get_capture_switch_visible():
    """True = everyone sees the Capture mode switch; False = everyone uses the default."""
    return get_setting("capture_switch", True) is not False


def _starting_branches():
    """The first list: an older list in settings.json, else the secret BRANCHES, else the default."""

    branches = load_settings().get("branches")
    if branches is None:
        branches = get_secret("BRANCHES", DEFAULT_BRANCHES)
    if isinstance(branches, str):                      # "Computer, IT" in the secrets
        branches = branches.split(",")

    return [str(b).strip() for b in branches if str(b).strip()]


def get_branches():
    """app4.py branches, from the database table "branches" (same list online and offline)."""

    names = _cached("branches", lambda store: store.get_branch_list())

    if names is None:                                  # database not reachable
        return _starting_branches()
    if names or get_setting("branches_seeded", False):
        return names                                   # (an emptied list stays empty)

    # first use of this database: fill the table once
    names = _starting_branches()
    save_branches(names)
    return names


def save_branches(branches):
    _, store = settings_store()
    store.save_branch_list(list(branches))
    store.save_app_setting("branches_seeded", True)
    _forget_cache()


def get_cooldown(role):
    """Seconds between Time In and Time Out for 'student' or 'teacher'."""

    try:
        return float(get_setting(f"cooldown_{role}", COOLDOWN_SECONDS))
    except (TypeError, ValueError):
        return COOLDOWN_SECONDS


def get_store(mode="offline"):
    if mode == "online":
        return SupabaseStore(*supabase_config())

    return SQLiteStore()


# ======================================================================
# Sync: copy what is missing from source to target (never deletes)
# ======================================================================

def sync_stores(source, target):
    """
    1. people missing in target are added (with their face embeddings)
       same ID but a different name on both sides = conflict -> skipped
       an empty year / division in target is filled in from source
    2. attendance (student + teacher tables): missing days are added; if both have
       the day, the one with a Time Out wins (it is more complete).
       The app4 evidence columns (IP, GPS, flag, review) are copied with the row.
    3. teacher classes and teacher accounts missing in target are added
    4. branches and App settings missing in target are added (nothing is overwritten)
    Returns what was copied.
    """

    result = {"people": 0, "embeddings": 0, "records": 0, "classes": 0, "accounts": 0,
              "branches": 0, "settings": 0, "conflicts": []}

    target_people = {p["id"]: p for p in target.get_people()}
    skipped = set()
    new_people = []

    # ---- 1. people ----
    for p in source.get_people():
        other = target_people.get(p["id"])

        if other is None:
            target.save_person(p["id"], p["name"], p["role"], p["registered_on"],
                               p.get("year"), p.get("division"))
            # keep a frozen account frozen and a waiting sign-up waiting
            if p.get("frozen"):
                target.set_frozen(p["id"], True)
            if p.get("approved") == 0:
                target.set_approved(p["id"], False)
            elif p.get("approved") == -1:
                target.reject_login(p["id"])
            new_people.append(p["id"])
            result["people"] += 1
        elif other["name"].strip().lower() != p["name"].strip().lower():
            skipped.add(p["id"])
            result["conflicts"].append(
                f"{p['id']} is '{p['name']}' here but '{other['name']}' on the other side")
        elif p.get("year") and not other.get("year"):
            target.set_class(p["id"], p["year"], p.get("division"))

    # faces only for the people just added (the others already have theirs)
    if new_people:
        ids, embs = source.get_all_embeddings()
        for person_id in new_people:
            mine = embs[ids == person_id]
            if len(mine):
                target.save_embeddings(person_id, mine)
                result["embeddings"] += len(mine)

    known = (set(target_people) | set(new_people)) - skipped

    # ---- 2. attendance ----
    target_records = {(r["person_id"], str(r["date"])): r for r in target.get_attendance()}

    for r in source.get_attendance():
        day = str(r["date"])
        if r["person_id"] not in known:
            continue

        other = target_records.get((r["person_id"], day))
        if other is None or (r.get("time_out") and not other.get("time_out")):
            target.save_record(r["person_id"], day, r.get("time_in"), r.get("time_out"))

            # app4.py evidence (IP, GPS readings, flag, teacher's decision) goes with the record
            if r.get("ip_address") or r.get("gps_readings") or r.get("flagged") or r.get("review"):
                target.save_check_info(r["person_id"], day, r.get("ip_address"),
                                       r.get("gps_readings"), bool(r.get("flagged")),
                                       r.get("flag_reason"))
                if r.get("review"):
                    target.set_review(r["person_id"], day, r["review"])

            result["records"] += 1

    # ---- 3. teacher classes + accounts ----
    target_classes = {(c["teacher_id"], c["year"], c["division"])
                      for c in target.get_teacher_classes()}
    missing = {}
    for c in source.get_teacher_classes():
        key = (c["teacher_id"], c["year"], c["division"])
        if key not in target_classes and c["teacher_id"] in known:
            missing.setdefault(c["teacher_id"], []).append((c["year"], c["division"]))

    for teacher_id, classes in missing.items():
        have = [(c["year"], c["division"]) for c in target.get_teacher_classes(teacher_id)]
        target.set_teacher_classes(teacher_id, have + classes)
        result["classes"] += len(classes)

    target_accounts = {a["username"]: a for a in target.get_teacher_accounts()}
    teachers_with_account = {a["person_id"] for a in target_accounts.values()}

    for a in source.get_teacher_accounts():
        if a["username"] in target_accounts or a["person_id"] not in known:
            continue

        # one account per teacher: don't add a second one
        if a["person_id"] in teachers_with_account:
            result["conflicts"].append(
                f"{a['person_id']} already has an account on the other side "
                f"('{a['username']}' not copied)")
            continue

        target.save_teacher_account(a["username"], a["password_hash"], a["person_id"])
        teachers_with_account.add(a["person_id"])
        result["accounts"] += 1

    # ---- 4. branches + App settings ----
    have = target.get_branch_list()
    have_lower = {b.lower() for b in have}
    new_branches = [b for b in source.get_branch_list() if b.lower() not in have_lower]
    if new_branches:
        target.save_branch_list(have + new_branches)
        result["branches"] = len(new_branches)

    target_settings = target.get_settings()
    for key, value in source.get_settings().items():
        if key not in target_settings:
            target.save_app_setting(key, value)
            result["settings"] += 1

    return result
