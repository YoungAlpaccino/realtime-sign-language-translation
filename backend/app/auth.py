"""JWT auth for WS connections and REST endpoints."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

# Dev default; override in production via env. HS256 symmetric signing.
SECRET_KEY = os.environ.get("SIGN_RT_JWT_SECRET", "dev-insecure-change-me")
ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = int(os.environ.get("SIGN_RT_JWT_TTL_MIN", "720"))


def create_token(subject: str, ttl_minutes: int | None = None) -> str:
    """Issue a signed JWT for a client/session."""
    ttl = TOKEN_TTL_MINUTES if ttl_minutes is None else ttl_minutes
    now = datetime.now(timezone.utc)
    payload = {"sub": subject, "iat": now, "exp": now + timedelta(minutes=ttl)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> str:
    """Validate a JWT and return the subject; raise ValueError on failure."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise ValueError(f"invalid token: {exc}") from exc
    sub = payload.get("sub")
    if not sub:
        raise ValueError("token missing subject")
    return str(sub)
