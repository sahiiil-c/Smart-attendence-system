"""
assets/location_checks.py - checks against fake GPS and attendance from outside college.

Kept separate from Streamlit (plain functions) so every check is easy to test and explain.

A "reading" is one GPS position from the browser:
    {"lat": 19.07, "lon": 72.87, "accuracy": 18.4, "time": 1761650000.0}
    accuracy = how sure the phone is, in metres (smaller = more precise)
    time     = seconds since 1970 (time.time())

The checks:
  1. MULTIPLE READINGS - 3 readings ~2 s apart. A real phone's GPS always "wobbles" a little;
     fake-GPS apps often return exactly the same numbers, or a round accuracy like 5 m.
  2. IMPOSSIBLE TRAVEL - from the last saved place and time we work out the speed needed
     to get here. Faster than 150 km/h is not possible on the way to college.
  3. SAME SPOT       - two students can't stand on exactly the same point (6 decimals = ~10 cm).
                       The same position usually means a shared fake location.
  4. COLLEGE WI-FI   - the internet address (IP) must be one of the college Wi-Fi addresses.
  5. GEOFENCE        - the average position must be inside the college circle.

decide() combines them: some failures BLOCK attendance, the suspicious ones only FLAG it
for a teacher to review.
"""

import ipaddress
import math

# -------- SETTINGS --------
READINGS_NEEDED = 3                  # GPS readings taken per check
SECONDS_BETWEEN_READINGS = 2         # gap between two readings (in the browser)
ROUND_ACCURACIES = {1, 3, 5, 10}     # accuracy values typical for fake-GPS apps
MAX_SPEED_KMH = 150                  # faster than this between two places = impossible
SAME_SPOT_MINUTES = 10               # "same spot" looks back this many minutes
SAME_SPOT_DECIMALS = 6               # lat/lon compared at 6 decimals (about 10 cm)
EARTH_RADIUS_M = 6_371_000

FLAG_TOO_PERFECT = "suspicious: too perfect"
FLAG_ROUND_ACCURACY = "suspicious: round accuracy"
FLAG_IMPOSSIBLE_TRAVEL = "suspicious: impossible travel"
FLAG_SAME_LOCATION = "suspicious: same location"


# ======================================================================
# Distances and the geofence
# ======================================================================

def distance_m(lat1, lon1, lat2, lon2):
    """Distance between two points on Earth in metres (haversine formula)."""

    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def average_position(readings):
    """The average of the readings -> (lat, lon). Used for the geofence check."""

    lat = sum(r["lat"] for r in readings) / len(readings)
    lon = sum(r["lon"] for r in readings) / len(readings)
    return lat, lon


def inside_geofence(lat, lon, center_lat, center_lon, radius_m):
    """True when (lat, lon) is inside the college circle."""
    return distance_m(lat, lon, center_lat, center_lon) <= radius_m


# ======================================================================
# Check 1: multiple readings
# ======================================================================

def too_perfect(readings):
    """All readings EXACTLY the same (lat, lon and accuracy) -> a real GPS never does that."""

    if len(readings) < 2:
        return False

    first = (readings[0]["lat"], readings[0]["lon"], readings[0]["accuracy"])
    return all((r["lat"], r["lon"], r["accuracy"]) == first for r in readings)


def round_accuracy(readings):
    """Accuracy is exactly 1, 3, 5 or 10 m on EVERY reading -> typical for fake-GPS apps."""

    return bool(readings) and all(r["accuracy"] in ROUND_ACCURACIES for r in readings)


# ======================================================================
# Check 2: impossible travel
# ======================================================================

def travel_speed_kmh(last_lat, last_lon, last_time, lat, lon, now_time):
    """Speed (km/h) needed to go from the last saved place to here in the time between."""

    hours = (now_time - last_time) / 3600
    km = distance_m(last_lat, last_lon, lat, lon) / 1000

    if hours <= 0:
        return 0.0 if km == 0 else float("inf")

    return km / hours


def impossible_travel(last, lat, lon, now_time):
    """
    last = the student's last saved {"lat", "lon", "time"} or None (first time).
    True when the speed needed is more than MAX_SPEED_KMH.
    """

    if not last:
        return False

    speed = travel_speed_kmh(last["lat"], last["lon"], last["time"], lat, lon, now_time)
    return speed > MAX_SPEED_KMH


# ======================================================================
# Check 3: same spot
# ======================================================================

def same_spot(person_id, lat, lon, now_time, recent):
    """
    recent = other students' saved positions: [{"person_id", "lat", "lon", "time"}, ...]
    Returns the IDs of OTHER students who were at exactly this point (6 decimals)
    in the last SAME_SPOT_MINUTES minutes.
    """

    here = (round(lat, SAME_SPOT_DECIMALS), round(lon, SAME_SPOT_DECIMALS))
    since = now_time - SAME_SPOT_MINUTES * 60

    return sorted({r["person_id"] for r in recent
                   if r["person_id"] != person_id
                   and r["time"] >= since
                   and (round(r["lat"], SAME_SPOT_DECIMALS),
                        round(r["lon"], SAME_SPOT_DECIMALS)) == here})


# ======================================================================
# Check 4: college Wi-Fi (IP address)
# ======================================================================

def client_ip(headers, fallback=None):
    """
    The visitor's internet address.
    Behind Cloudflare / a proxy (Streamlit Cloud) the real address is in a header:
      1. CF-Connecting-IP
      2. the FIRST address in X-Forwarded-For ("client, proxy1, proxy2")
      3. otherwise the address of the connection itself (fallback)
    """

    headers = {str(k).lower(): v for k, v in (headers or {}).items()}

    if headers.get("cf-connecting-ip"):
        return headers["cf-connecting-ip"].strip()

    if headers.get("x-forwarded-for"):
        return headers["x-forwarded-for"].split(",")[0].strip()

    return fallback


def ip_allowed(ip, allowed):
    """
    allowed = list of college addresses: single IPs ("203.0.113.7")
              or ranges in CIDR form ("203.0.113.0/24" = 203.0.113.0 - 203.0.113.255).
    """

    try:
        address = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return False

    for entry in allowed:
        try:
            if address in ipaddress.ip_network(str(entry).strip(), strict=False):
                return True
        except ValueError:
            continue                     # a typo in the list is skipped, not a crash

    return False


# ======================================================================
# The decision
# ======================================================================

def decide(inside, face_ok, wifi_required, ip_ok, flags):
    """
    BLOCK (nothing saved) when:
      * outside the college geofence, or
      * the face was not recognised, or
      * the college Wi-Fi is required and this IP is not a college address.
    FLAG (saved, but a teacher must Approve/Reject) when any suspicious check fired.
    Otherwise OK.
    Returns ("blocked", reason) / ("flagged", [reasons]) / ("ok", []).
    """

    if not inside:
        return "blocked", "You are outside the college area."
    if not face_ok:
        return "blocked", "Face not recognised."
    if wifi_required and not ip_ok:
        return "blocked", "Please connect to the college Wi-Fi."
    if flags:
        return "flagged", sorted(set(flags))
    return "ok", []


def gps_flags(readings, last, recent, person_id, now_time):
    """
    All suspicious GPS checks for one student at once.
    Returns (flags, other_ids): the flag texts and the other students on the same spot.
    """

    lat, lon = average_position(readings)
    flags = []

    if too_perfect(readings):
        flags.append(FLAG_TOO_PERFECT)
    if round_accuracy(readings):
        flags.append(FLAG_ROUND_ACCURACY)
    if impossible_travel(last, lat, lon, now_time):
        flags.append(FLAG_IMPOSSIBLE_TRAVEL)

    others = same_spot(person_id, lat, lon, now_time, recent)
    if others:
        flags.append(FLAG_SAME_LOCATION)

    return flags, others
