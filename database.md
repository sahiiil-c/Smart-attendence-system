# Putting the database online (Supabase) – step by step

Right now everything is saved in the local file `data.db` (SQLite). It upgrades itself, so it needs
nothing from you. To use the app **online** (Streamlit Cloud) the data must live in **Supabase**,
because Streamlit Cloud deletes local files whenever the app restarts.

The app to deploy is **`app4.py`**. (`app.py` and `app2.py` are earlier versions; all three use the
same database, so this guide works for them too.)

Do these steps **once, when the project is finished**:

| # | Step | Where |
|---|---|---|
| 1 | Create a Supabase project | supabase.com |
| 2 | Create the tables (**Part A** SQL below) | Supabase → SQL Editor |
| 3 | Put the Supabase keys in the secrets (correct layout) | `.streamlit/secrets.toml` |
| 4 | Upload your local data and switch the app to Online | the app → Admin → 🗄 Database |
| 5 | Deploy and paste the same secrets | share.streamlit.io |
| 6 | Check that everything arrived | the online app |

---

## Step 1 – create the Supabase project
1. <https://supabase.com> → **Start your project** → sign in with GitHub.
2. **New project** → name it, set a database password (save it), region closest to you (e.g. Mumbai)
   → **Create new project**. Wait ~2 minutes.

## Step 2 – create the tables
Left menu → **SQL Editor** → **New query** → paste → **Run**.

* **New, empty project → Part A** (creates everything every app uses).
* **You already created the older tables** (from an earlier README) → **Part B** instead.
  It upgrades them and keeps every row (nothing is deleted).

### Part A – new project (everything)

```sql
create table people (
  id            text primary key,           -- ID_001 / T_001
  name          text not null,
  role          text not null,              -- student / teacher
  registered_on date not null,
  year          text,                       -- FE / SE / TE / BE (students)
  division      text,                       -- app4.py: the BRANCH (e.g. Computer); app2.py: A-D
  frozen        integer not null default 0, -- 1 = frozen (face used in a teacher sign-up)
  approved      integer not null default 1  -- app4.py: 1 may log in, 0 waiting, -1 login rejected
);

create table embeddings (
  id        bigint generated always as identity primary key,
  person_id text not null references people(id) on delete cascade,
  embedding text not null                   -- 512 float32 numbers, base64 encoded
);

create table student_attendance (
  person_id    text not null references people(id) on delete cascade,
  date         date not null,
  time_in      text,
  time_out     text,
  ip_address   text,                        -- app4.py evidence
  gps_readings text,                        -- JSON: {"saved_at": ..., "readings": [3 readings]}
  flagged      integer not null default 0,  -- 1 = suspicious, a teacher must review it
  flag_reason  text,                        -- e.g. "suspicious: too perfect"
  review       text,                        -- '' / approved / rejected
  primary key (person_id, date)
);

create table teacher_attendance (
  person_id    text not null references people(id) on delete cascade,
  date         date not null,
  time_in      text,
  time_out     text,
  ip_address   text,
  gps_readings text,
  flagged      integer not null default 0,
  flag_reason  text,
  review       text,
  primary key (person_id, date)
);

create table teacher_classes (
  teacher_id text not null references people(id) on delete cascade,
  year       text not null,
  division   text not null,
  primary key (teacher_id, year, division)
);

create table teacher_accounts (
  username      text primary key,
  password_hash text not null,              -- salted PBKDF2 hash, never the real password
  person_id     text not null references people(id) on delete cascade
);

-- one login account per teacher ID
create unique index if not exists teacher_accounts_one_per_teacher
  on teacher_accounts (person_id);

create table login_sessions (
  token_hash text primary key,              -- SHA-256 of the cookie token (never the token itself)
  role       text not null,                 -- Admin / Teacher / Student
  person_id  text not null,                 -- ID_001 / T_001 / ADMIN
  expires_at text not null                  -- after this time you must log in again
);

create table alerts (
  id         bigint generated always as identity primary key,
  created_at text not null,                 -- when it happened (local time)
  student_id text not null,                 -- the frozen student
  message    text not null,                 -- what happened
  status     text not null default 'open'   -- open / restored / deleted
);

create table branches (
  name     text primary key,                -- e.g. Computer (app4.py branch list)
  position integer not null default 0       -- order in the list
);

create table app_settings (
  key   text primary key,                   -- e.g. cooldown_student, allowed_ips
  value text                                -- the value as JSON text
);

-- Row Level Security ON with no policies = the public "anon" key can read nothing.
-- The app uses the secret service_role key, which is allowed through.
alter table people             enable row level security;
alter table embeddings         enable row level security;
alter table student_attendance enable row level security;
alter table teacher_attendance enable row level security;
alter table teacher_classes    enable row level security;
alter table teacher_accounts   enable row level security;
alter table login_sessions     enable row level security;
alter table alerts             enable row level security;
alter table branches           enable row level security;
alter table app_settings       enable row level security;
```

### Part B – upgrade older tables (keeps all data, safe to run again)

```sql
-- 1. new columns on people (everybody already there stays approved)
alter table people add column if not exists year     text;
alter table people add column if not exists division text;
alter table people add column if not exists frozen   integer not null default 0;
alter table people add column if not exists approved integer not null default 1;

-- 2. the newer tables
create table if not exists student_attendance (
  person_id text not null references people(id) on delete cascade,
  date      date not null,
  time_in   text,
  time_out  text,
  primary key (person_id, date)
);

create table if not exists teacher_attendance (
  person_id text not null references people(id) on delete cascade,
  date      date not null,
  time_in   text,
  time_out  text,
  primary key (person_id, date)
);

create table if not exists teacher_classes (
  teacher_id text not null references people(id) on delete cascade,
  year       text not null,
  division   text not null,
  primary key (teacher_id, year, division)
);

create table if not exists teacher_accounts (
  username      text primary key,
  password_hash text not null,
  person_id     text not null references people(id) on delete cascade
);

-- one login account per teacher ID.
-- If this line fails, a teacher already has 2+ accounts: find them with
--   select person_id, count(*) from teacher_accounts group by person_id having count(*) > 1;
-- delete the extra account in Admin -> Teacher accounts, then run the line again.
create unique index if not exists teacher_accounts_one_per_teacher
  on teacher_accounts (person_id);

create table if not exists login_sessions (
  token_hash text primary key,
  role       text not null,
  person_id  text not null,
  expires_at text not null
);

create table if not exists alerts (
  id         bigint generated always as identity primary key,
  created_at text not null,
  student_id text not null,
  message    text not null,
  status     text not null default 'open'
);

create table if not exists branches (
  name     text primary key,
  position integer not null default 0
);

create table if not exists app_settings (
  key   text primary key,
  value text
);

-- 3. app4.py evidence columns on BOTH attendance tables
alter table student_attendance add column if not exists ip_address   text;
alter table student_attendance add column if not exists gps_readings text;
alter table student_attendance add column if not exists flagged      integer not null default 0;
alter table student_attendance add column if not exists flag_reason  text;
alter table student_attendance add column if not exists review       text;

alter table teacher_attendance add column if not exists ip_address   text;
alter table teacher_attendance add column if not exists gps_readings text;
alter table teacher_attendance add column if not exists flagged      integer not null default 0;
alter table teacher_attendance add column if not exists flag_reason  text;
alter table teacher_attendance add column if not exists review       text;

alter table student_attendance enable row level security;
alter table teacher_attendance enable row level security;
alter table teacher_classes    enable row level security;
alter table teacher_accounts   enable row level security;
alter table login_sessions     enable row level security;
alter table alerts             enable row level security;
alter table branches           enable row level security;
alter table app_settings       enable row level security;

-- 4. an old single "attendance" table: copy its rows by role, keep it as attendance_backup
do $$
begin
  if to_regclass('public.attendance') is not null then

    insert into student_attendance (person_id, date, time_in, time_out)
    select a.person_id, a.date, a.time_in, a.time_out
    from attendance a join people p on p.id = a.person_id
    where p.role = 'student'
    on conflict (person_id, date) do nothing;

    insert into teacher_attendance (person_id, date, time_in, time_out)
    select a.person_id, a.date, a.time_in, a.time_out
    from attendance a join people p on p.id = a.person_id
    where p.role = 'teacher'
    on conflict (person_id, date) do nothing;

    alter table attendance rename to attendance_backup;
  end if;
end $$;
```

`flagged` and `frozen` are integers 0/1, and `approved` is 1 (may log in) / 0 (waiting for approval) /
-1 (login rejected). They are integers (not `boolean`) so they mean exactly the same in SQLite and
Supabase.

### Check the tables (all counts are 0 in a new project)

```sql
select 'people' as table_name, count(*) from people
union all select 'embeddings',         count(*) from embeddings
union all select 'student_attendance', count(*) from student_attendance
union all select 'teacher_attendance', count(*) from teacher_attendance
union all select 'teacher_classes',    count(*) from teacher_classes
union all select 'teacher_accounts',   count(*) from teacher_accounts
union all select 'login_sessions',     count(*) from login_sessions
union all select 'alerts',             count(*) from alerts
union all select 'branches',           count(*) from branches
union all select 'app_settings',       count(*) from app_settings;
```

---

## Step 3 – the keys in the secrets (this layout matters!)

Supabase → **Project Settings → API** (or **API Keys**):
* **Project URL** → `SUPABASE_URL`
* **service_role** key (or the new **secret** key `sb_secret_...`) → `SUPABASE_KEY`
  ⚠ This key can read and write everything: only in the secrets, never in code or on GitHub.

Replace the whole `.streamlit/secrets.toml` with this layout. **No `[supabase]` line** – a line in
square brackets makes every line under it part of a section, so the app can't see your admin password
(see `suggestion.md`):

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-service-role-or-secret-key"
STORAGE_MODE = "online"            # Streamlit Cloud starts on the online database by itself

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "a-strong-password"

# app4.py – college geofence + Wi-Fi (starting values; changeable in App settings)
GPS_REQUIRED = true
COLLEGE_LAT = 19.076000
COLLEGE_LON = 72.877700
GEOFENCE_RADIUS_M = 200
ALLOWED_IPS = ["203.0.113.0/24"]
WIFI_REQUIRED = false
```

Restart the app after saving (Ctrl+C, then `streamlit run app4.py`). Your admin login is now the one
above – `admin123` stops working.

## Step 4 – upload your local data, then switch to Online
On your laptop (`streamlit run app4.py`), log in as admin (Teacher login with the admin username +
password) → **🗄 Database**:

1. **⬆ Upload offline → online** – copies people (with branch, frozen state and approval state:
   waiting / approved / login rejected), face data, student + teacher attendance (with the app4
   IP / GPS / flag / review evidence), teacher classes, teacher accounts, the **branch list** and all
   **App settings** (capture mode, cooldowns, GPS on/off, college centre, ALLOWED_IPS, …).
   It never deletes or overwrites anything; run it again any time, it only adds what's missing.
   People are matched **by ID**: the same ID with a different name on the two sides is a conflict
   (skipped and shown as a warning).
2. **View data → ☁️ Online** – compare each table with **💻 Offline** (same number of rows).
3. **Storage in use → ☁️ Online** – from now on the whole app saves to Supabase.

On your laptop the choice you make here (saved in `settings.json`) wins over `STORAGE_MODE`.

## Step 5 – deploy on Streamlit Cloud
1. Push the project to GitHub (`.gitignore` already keeps `secrets.toml`, `data.db`, `settings.json`,
   `login.md` and the face model out of it).
2. <https://share.streamlit.io> → **Create app** → your repo, branch `main`, main file **`app4.py`**.
3. **Advanced settings → Secrets**: paste the same text as in step 3. **Python version**: the newest
   offered (your `requirements.txt` was made with Python 3.14 on your laptop).
4. **Deploy**. The first start takes several minutes (packages + face model download).

## Step 6 – after the first online start
* Open the deployed app, log in as admin → **🗄 Database → View data → ☁️ Online**: people, branches
  and App settings should be there. The sidebar shows "Database: Supabase (online database)".
* Everybody logs in again once (login sessions are not uploaded).
* **App settings and branches are in the database** (tables `app_settings`, `branches`), so the online
  app has exactly your settings and branch list, on every device and after every restart. Only the
  choice offline / online itself stays in `settings.json` (online: the `STORAGE_MODE` secret).
* The 🚨 Alerts history stays local (the frozen state itself is uploaded).

## What is not uploaded
| Not uploaded | Why / what to do |
|---|---|
| `login_sessions` | security: people simply log in again |
| `alerts` (frozen-student history) | the frozen state itself **is** uploaded; decide open alerts before uploading |
| `settings.json` | it only holds the offline/online choice now (online: the `STORAGE_MODE` secret) |
| `attendance_backup`, `data_backup_before_upgrade.db` | old backups, not used any more – delete them only when you are sure |

## If something goes wrong
| You see | Meaning | Fix |
|---|---|---|
| `relation "people" does not exist` / `student_attendance` missing | tables not created | run Part A (new) or Part B (old tables) |
| `column ... does not exist` (e.g. `approved`, `gps_readings`) | older tables | run Part B again – it only adds what's missing |
| `Invalid API key` / 401 | wrong key | copy the **service_role / secret** key again (not the anon key) |
| Online can't be chosen in 🗄 Database | the app can't see the keys | check the layout in step 3 (no `[supabase]` line) and restart |
| Admin password doesn't work | ADMIN_ lines under a `[section]` | layout in step 3 |
| `Could not find a version that satisfies ...` on Streamlit Cloud | that Python version is too old for a pinned package | choose a newer Python in Settings, then reboot |
| `ImportError: libGL.so.1` | OpenCV needs a Linux library | `packages.txt` must contain `libgl1` and `libglib2.0-0` |

---

## What the tables hold

| Table | What it holds |
|---|---|
| `people` | everyone: `id` (ID_001 / T_001), `name`, `role`, `registered_on`, `year`, `division` (**app4.py: the branch**, e.g. Computer), `frozen` (0/1), `approved` (1 may log in / 0 waiting / -1 login rejected) |
| `embeddings` | face data (several rows per person) |
| `student_attendance` | one row per **student** per day + app4 evidence (`ip_address`, `gps_readings`, `flagged`, `flag_reason`, `review`) |
| `teacher_attendance` | one row per **teacher** per day (same columns) |
| `teacher_classes` | which classes (year + division) a teacher teaches; one class can have many teachers |
| `teacher_accounts` | teacher logins: username + hashed password – **one account per teacher ID** |
| `alerts` | frozen students waiting for the admin's decision (Restore / Delete) and the history |
| `login_sessions` | "stay logged in" after a page reload: only a hash of the browser cookie token, plus an expiry date |
| `branches` | app4.py branch list (name + order), edited in Admin → App settings → 🏫 Branches |
| `app_settings` | every admin App setting (one row per setting, value as JSON) – the same for everybody, synced online |
