"""
Session-cookie authentication for the browser UI.

The API-key machinery in `security.py` is machine-to-machine. This module adds
a login form + signed session cookie so the viewer can be exposed publicly
without sitting behind an external access proxy.

Passwords are stored as PBKDF2-SHA256 hashes in CCTV_AUTH_USERS:

    CCTV_AUTH_USERS="admin:pbkdf2_sha256$260000$<salt_b64>$<hash_b64>"

Generate a hash with:  python -m backend.middleware.session_auth <password>
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse

COOKIE_NAME = "cctv_session"
DEFAULT_ITERATIONS = 260_000
# How long a session stays valid, in seconds (default 12h).
SESSION_TTL = int(os.environ.get("CCTV_SESSION_TTL", str(12 * 60 * 60)))

# Brute-force throttle: max failed logins per IP within the window.
MAX_FAILED_ATTEMPTS = 8
FAILED_WINDOW = 300.0


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Hash a password into the storable pbkdf2_sha256 format."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode(),
        base64.b64encode(digest).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a password against a stored pbkdf2 hash."""
    try:
        algo, iter_s, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), base64.b64decode(salt_b64), int(iter_s)
        )
        return hmac.compare_digest(digest, base64.b64decode(hash_b64))
    except (ValueError, TypeError):
        return False


def get_users() -> Dict[str, str]:
    """Parse CCTV_AUTH_USERS into {username: password_hash}."""
    raw = os.environ.get("CCTV_AUTH_USERS", "").strip()
    if not raw:
        return {}
    users: Dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # Only split on the first colon; the hash itself contains none, but be safe.
        username, _, pw_hash = entry.partition(":")
        if username and pw_hash:
            # Docker Compose escapes '$' as '$$' during interpolation. A real
            # pbkdf2 hash is base64 + '$' separators and never contains '$$',
            # so collapsing it back is unambiguous.
            users[username.strip()] = pw_hash.strip().replace("$$", "$")
    return users


# ---------------------------------------------------------------------------
# Session tokens
# ---------------------------------------------------------------------------

def _secret() -> bytes:
    """Signing secret. Generated per-process if unset (sessions drop on restart)."""
    configured = os.environ.get("CCTV_SESSION_SECRET", "").strip()
    if configured:
        return configured.encode()
    global _EPHEMERAL_SECRET
    if _EPHEMERAL_SECRET is None:
        _EPHEMERAL_SECRET = secrets.token_bytes(32)
    return _EPHEMERAL_SECRET


_EPHEMERAL_SECRET: Optional[bytes] = None


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64d(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def create_session_token(username: str, ttl: int = SESSION_TTL) -> str:
    """Build a signed `payload.signature` session token."""
    payload = _b64e(json.dumps({"u": username, "exp": int(time.time()) + ttl}).encode())
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    return f"{payload}.{_b64e(sig)}"


def verify_session_token(token: Optional[str]) -> Optional[str]:
    """Return the username for a valid, unexpired token, else None."""
    if not token or "." not in token:
        return None
    payload, _, sig = token.rpartition(".")
    expected = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(expected, _b64d(sig)):
            return None
        data = json.loads(_b64d(payload))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("exp", 0) < time.time():
        return None
    username = data.get("u")
    return username if isinstance(username, str) else None


# ---------------------------------------------------------------------------
# Login throttling
# ---------------------------------------------------------------------------

_failed_attempts: Dict[str, List[float]] = defaultdict(list)


def record_failed_login(client_ip: str) -> None:
    _failed_attempts[client_ip].append(time.time())


def is_login_throttled(client_ip: str) -> bool:
    """True when this IP has burned through its failed-login budget."""
    cutoff = time.time() - FAILED_WINDOW
    attempts = [t for t in _failed_attempts[client_ip] if t > cutoff]
    _failed_attempts[client_ip] = attempts
    return len(attempts) >= MAX_FAILED_ATTEMPTS


def clear_failed_logins(client_ip: str) -> None:
    _failed_attempts.pop(client_ip, None)


def authenticate(username: str, password: str) -> bool:
    """Check credentials, spending equal time on unknown and known usernames."""
    users = get_users()
    stored = users.get(username)
    if stored is None:
        # Dummy verify so a bad username costs the same as a bad password.
        verify_password(password, hash_password("dummy", iterations=1000))
        return False
    return verify_password(password, stored)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class SessionAuthMiddleware(BaseHTTPMiddleware):
    """Require a valid session cookie, redirecting browsers to the login page."""

    def __init__(self, app, exclude_paths: Optional[List[str]] = None,
                 exclude_prefixes: Optional[List[str]] = None, enabled: bool = True,
                 login_path: str = "/login.html"):
        super().__init__(app)
        self.enabled = enabled
        self.login_path = login_path
        self.exclude_paths = set(exclude_paths or [])
        self.exclude_prefixes = exclude_prefixes or []

    def _is_excluded(self, path: str) -> bool:
        if path in self.exclude_paths:
            return True
        return any(path.startswith(prefix) for prefix in self.exclude_prefixes)

    async def dispatch(self, request, call_next):
        if not self.enabled or not get_users():
            return await call_next(request)

        if self._is_excluded(request.url.path):
            return await call_next(request)

        username = verify_session_token(request.cookies.get(COOKIE_NAME))
        if not username:
            # Navigations get a redirect; XHR/fetch gets a 401 to handle in JS.
            accepts_html = "text/html" in request.headers.get("accept", "")
            if accepts_html and request.method == "GET":
                nxt = request.url.path
                if request.url.query:
                    nxt += "?" + request.url.query
                return RedirectResponse(
                    f"{self.login_path}?next={base64.urlsafe_b64encode(nxt.encode()).decode()}",
                    status_code=302,
                )
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "message": "Login required"},
            )

        request.state.username = username
        return await call_next(request)


class NoStoreMiddleware(BaseHTTPMiddleware):
    """Stop HTML and API responses being cached.

    Without this, a browser (or any intermediary) can serve a logged-in page, or
    a stale pre-login redirect, to someone who should have been sent to the login
    form. Static assets under /styles and /scripts stay cacheable.
    """

    CACHEABLE_PREFIXES = ("/styles/", "/scripts/")

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(self.CACHEABLE_PREFIXES):
            return response
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type or "application/json" in content_type \
                or response.status_code in (301, 302, 303, 307, 308):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


def cookie_kwargs() -> dict:
    """Cookie flags. Secure is on unless explicitly disabled for local HTTP."""
    secure = os.environ.get("CCTV_COOKIE_SECURE", "true").lower() == "true"
    return {
        "httponly": True,
        "secure": secure,
        "samesite": "lax",
        "path": "/",
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m backend.middleware.session_auth <password>")
        raise SystemExit(1)
    print(hash_password(sys.argv[1]))
