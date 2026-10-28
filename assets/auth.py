"""
assets/auth.py - passwords and the admin login.

Teacher passwords are never stored as plain text. We store a "hash":
  * a random salt (so two equal passwords give different hashes)
  * PBKDF2-SHA256 run 100,000 times (slow on purpose, so guessing is slow)
Stored format: "<salt hex>$<hash hex>"

"Stay logged in" (app2.py): after a login the browser gets a random token in a cookie.
The database keeps only the SHA-256 hash of that token, so a copy of the database
can't be used to log in. The token is not the password.
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from assets import storage

ITERATIONS = 100_000
LOGIN_DAYS = 7            # how long you stay logged in


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def start_login(store, user):
    """Remember a login. user = {"role", "id", "name"}. Returns the token for the cookie."""

    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=LOGIN_DAYS)
    store.save_login_session(_token_hash(token), user["role"], user["id"], expires.isoformat())

    return token


def user_from_token(store, token):
    """The logged-in user for a cookie token, or None (unknown, expired, person deleted)."""

    if not token:
        return None

    session = store.get_login_session(_token_hash(token))

    if not session:
        return None

    if datetime.fromisoformat(session["expires_at"]) < datetime.now(timezone.utc):
        store.delete_login_session(_token_hash(token))
        return None

    if session["role"] == "Admin":
        return {"role": "Admin", "id": "ADMIN", "name": "Admin"}

    person = store.get_person(session["person_id"])

    if not person or person["role"] != session["role"].lower() or person.get("frozen"):
        return None

    if person.get("approved") in (0, -1):      # app4.py: waiting for approval / login rejected
        return None

    return {"role": session["role"], "id": person["id"], "name": person["name"]}


def end_login(store, token):
    if token:
        store.delete_login_session(_token_hash(token))


def hash_password(password):

    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)

    return f"{salt.hex()}${digest.hex()}"


def check_password(password, stored):

    try:
        salt_hex, digest_hex = stored.split("$")
    except (AttributeError, ValueError):
        return False

    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), ITERATIONS)

    # compare_digest takes the same time for right and wrong answers (safer)
    return hmac.compare_digest(digest.hex(), digest_hex)


def is_admin_username(username):
    """True for the admin's username - teachers may not use it."""
    return username.strip().lower() == str(storage.get_secret("ADMIN_USERNAME", "admin")).strip().lower()


def check_admin(username, password):
    """Admin username/password come from st.secrets (defaults admin / admin123)."""

    good_user = storage.get_secret("ADMIN_USERNAME", "admin")
    good_pass = storage.get_secret("ADMIN_PASSWORD", "admin123")

    # compare bytes (compare_digest refuses non-ASCII text)
    return (hmac.compare_digest(username.strip().encode(), str(good_user).encode())
            and hmac.compare_digest(password.encode(), str(good_pass).encode()))
