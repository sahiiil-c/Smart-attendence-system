"""
tests/conftest.py - runs before EVERY test.

App settings and branches are stored in the database now, so a test that changes a setting
would otherwise write into the real data.db / settings.json - or even reach the real
Supabase project through the keys in .streamlit/secrets.toml. This:
  * points every test to temporary files,
  * hides the real secrets (no Supabase keys, no admin password) from storage.get_secret,
  * turns the settings cache off (tests change settings between runs).
A test that needs a secret patches storage.get_secret itself.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def keep_real_files_safe(tmp_path, monkeypatch):
    db_path = str(tmp_path / "safe.db")
    settings_path = str(tmp_path / "safe_settings.json")

    monkeypatch.setenv("ATTENDANCE_DB", db_path)
    monkeypatch.setenv("ATTENDANCE_SETTINGS", settings_path)
    monkeypatch.setenv("ATTENDANCE_SETTINGS_CACHE", "0")

    from assets import storage                      # the copy that is loaded right now
    monkeypatch.setattr(storage, "DB_PATH", db_path)
    monkeypatch.setattr(storage, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(storage, "SETTINGS_CACHE_SECONDS", 0)

    # never the real secrets.toml: no Supabase keys -> the real online database can't be reached
    monkeypatch.setattr(storage, "get_secret", lambda name, default=None: default)
    monkeypatch.setattr(storage, "supabase_config", lambda: (None, None))
    storage._settings_stores.clear()
    storage._settings_cache.clear()
