"""
Tests for browser session authentication (login form + signed cookie).
"""
import base64
import json
import time

import pytest

from middleware import session_auth


# ============================================================================
# Password hashing
# ============================================================================

class TestPasswordHashing:

    def test_roundtrip(self):
        stored = session_auth.hash_password("correct horse battery staple")
        assert session_auth.verify_password("correct horse battery staple", stored)

    def test_wrong_password_rejected(self):
        stored = session_auth.hash_password("hunter2")
        assert not session_auth.verify_password("hunter3", stored)

    def test_salt_is_random(self):
        """Same password must not produce the same hash twice."""
        assert session_auth.hash_password("same") != session_auth.hash_password("same")

    @pytest.mark.parametrize("malformed", ["", "notahash", "md5$1$a$b", "pbkdf2_sha256$x"])
    def test_malformed_hash_rejected(self, malformed):
        assert not session_auth.verify_password("anything", malformed)


# ============================================================================
# User parsing
# ============================================================================

class TestUserParsing:

    def test_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("CCTV_AUTH_USERS", raising=False)
        assert session_auth.get_users() == {}

    def test_parses_multiple_users(self, monkeypatch):
        h1 = session_auth.hash_password("a")
        h2 = session_auth.hash_password("b")
        monkeypatch.setenv("CCTV_AUTH_USERS", f"alice:{h1},bob:{h2}")
        users = session_auth.get_users()
        assert set(users) == {"alice", "bob"}

    def test_authenticate(self, monkeypatch):
        h = session_auth.hash_password("s3cret")
        monkeypatch.setenv("CCTV_AUTH_USERS", f"admin:{h}")
        assert session_auth.authenticate("admin", "s3cret")
        assert not session_auth.authenticate("admin", "wrong")
        assert not session_auth.authenticate("nobody", "s3cret")


# ============================================================================
# Session tokens
# ============================================================================

class TestSessionTokens:

    @pytest.fixture(autouse=True)
    def _secret(self, monkeypatch):
        monkeypatch.setenv("CCTV_SESSION_SECRET", "unit-test-secret")

    def test_roundtrip(self):
        token = session_auth.create_session_token("admin")
        assert session_auth.verify_session_token(token) == "admin"

    def test_expired_token_rejected(self):
        assert session_auth.verify_session_token(
            session_auth.create_session_token("admin", ttl=-1)
        ) is None

    def test_tampered_signature_rejected(self):
        token = session_auth.create_session_token("admin")
        payload, _, sig = token.rpartition(".")
        assert session_auth.verify_session_token(f"{payload}.{sig[:-4]}AAAA") is None

    def test_tampered_payload_rejected(self):
        """Re-encoding the payload as a different user must not validate."""
        token = session_auth.create_session_token("admin")
        _, _, sig = token.rpartition(".")
        forged = base64.urlsafe_b64encode(
            json.dumps({"u": "root", "exp": int(time.time()) + 999}).encode()
        ).decode().rstrip("=")
        assert session_auth.verify_session_token(f"{forged}.{sig}") is None

    def test_unsigned_token_rejected(self):
        """A payload with no signature at all must not be accepted."""
        forged = base64.urlsafe_b64encode(
            json.dumps({"u": "root", "exp": int(time.time()) + 999}).encode()
        ).decode().rstrip("=")
        assert session_auth.verify_session_token(forged) is None

    def test_token_from_a_different_secret_rejected(self, monkeypatch):
        token = session_auth.create_session_token("admin")
        monkeypatch.setenv("CCTV_SESSION_SECRET", "a-different-secret")
        assert session_auth.verify_session_token(token) is None

    @pytest.mark.parametrize("garbage", [None, "", "no-dot", "...", "a.b"])
    def test_garbage_rejected(self, garbage):
        assert session_auth.verify_session_token(garbage) is None


# ============================================================================
# Login throttling
# ============================================================================

class TestLoginThrottle:

    @pytest.fixture(autouse=True)
    def _clean(self):
        session_auth._failed_attempts.clear()
        yield
        session_auth._failed_attempts.clear()

    def test_not_throttled_initially(self):
        assert not session_auth.is_login_throttled("1.2.3.4")

    def test_throttles_after_budget_spent(self):
        for _ in range(session_auth.MAX_FAILED_ATTEMPTS):
            session_auth.record_failed_login("1.2.3.4")
        assert session_auth.is_login_throttled("1.2.3.4")

    def test_throttle_is_per_ip(self):
        for _ in range(session_auth.MAX_FAILED_ATTEMPTS):
            session_auth.record_failed_login("1.2.3.4")
        assert not session_auth.is_login_throttled("5.6.7.8")

    def test_success_clears_the_counter(self):
        for _ in range(session_auth.MAX_FAILED_ATTEMPTS):
            session_auth.record_failed_login("1.2.3.4")
        session_auth.clear_failed_logins("1.2.3.4")
        assert not session_auth.is_login_throttled("1.2.3.4")

    def test_old_attempts_fall_out_of_the_window(self):
        stale = time.time() - session_auth.FAILED_WINDOW - 1
        session_auth._failed_attempts["1.2.3.4"] = [stale] * session_auth.MAX_FAILED_ATTEMPTS
        assert not session_auth.is_login_throttled("1.2.3.4")


class TestComposeEscaping:
    """Docker Compose escapes '$' as '$$'; the parser must tolerate both forms."""

    def test_double_dollar_hash_is_normalized(self, monkeypatch):
        h = session_auth.hash_password("s3cret")
        monkeypatch.setenv("CCTV_AUTH_USERS", "admin:" + h.replace("$", "$$"))
        assert session_auth.authenticate("admin", "s3cret")

    def test_plain_hash_still_works(self, monkeypatch):
        h = session_auth.hash_password("s3cret")
        monkeypatch.setenv("CCTV_AUTH_USERS", f"admin:{h}")
        assert session_auth.authenticate("admin", "s3cret")
