import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
from uuid import UUID, uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, Request
from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from app.api_errors import api_error
from app.database import get_database_session
from app.models import AnonymousIdentity, Person, RecognitionProcess, User


ACCESS_COOKIE = "face_access"
REFRESH_COOKIE = "face_refresh"
ACCESS_MINUTES = max(5, int(os.getenv("AUTH_ACCESS_MINUTES", "30")))
REFRESH_DAYS = max(1, int(os.getenv("AUTH_REFRESH_DAYS", "7")))
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}
AUTH_SECRET = os.getenv("AUTH_SECRET_KEY", "")
if len(AUTH_SECRET) < 32:
    raise RuntimeError("AUTH_SECRET_KEY must contain at least 32 characters.")

_password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except (InvalidHashError, VerifyMismatchError):
        return False


def encode_access_token(user: User, session_id: UUID) -> Tuple[str, datetime]:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_MINUTES)
    header = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64encode(json.dumps({
        "sub": str(user.id),
        "sid": str(session_id),
        "role": user.role,
        "typ": "access",
        "exp": int(expires_at.timestamp()),
    }, separators=(",", ":")).encode())
    message = f"{header}.{payload}"
    signature = _b64encode(hmac.new(AUTH_SECRET.encode(), message.encode(), hashlib.sha256).digest())
    return f"{message}.{signature}", expires_at


def decode_access_token(token: str) -> Dict[str, object]:
    try:
        header, payload, signature = token.split(".")
        message = f"{header}.{payload}"
        expected = _b64encode(hmac.new(AUTH_SECRET.encode(), message.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        claims = json.loads(_b64decode(payload))
        if claims.get("typ") != "access" or int(claims["exp"]) <= int(datetime.now(timezone.utc).timestamp()):
            raise ValueError("expired")
        UUID(str(claims["sub"]))
        UUID(str(claims["sid"]))
        return claims
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise api_error(401, "AUTH_INVALID", "Oturum gecersiz veya suresi dolmus.") from error


def refresh_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def current_user_id_from_request(request: Request) -> Optional[UUID]:
    token = request.cookies.get(ACCESS_COOKIE)
    if not token:
        return None
    try:
        return UUID(str(decode_access_token(token)["sub"]))
    except Exception:
        return None


def get_current_user(
    request: Request,
    session: Session = Depends(get_database_session),
) -> User:
    token = request.cookies.get(ACCESS_COOKIE)
    if not token:
        raise api_error(401, "AUTH_REQUIRED", "Bu islem icin giris yapmalisiniz.")
    claims = decode_access_token(token)
    user = session.get(User, UUID(str(claims["sub"])))
    if user is None or not user.is_active:
        raise api_error(401, "AUTH_INACTIVE", "Kullanici hesabi aktif degil.")
    request.state.current_user = user
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise api_error(403, "ADMIN_REQUIRED", "Bu islem yalnizca yoneticilere aciktir.")
    return user


def ensure_initial_admin(session: Session) -> User:
    username = os.getenv("ADMIN_INITIAL_USERNAME", "admin").strip().lower()
    email = os.getenv("ADMIN_INITIAL_EMAIL", "admin@yuztanima.local").strip().lower()
    full_name = os.getenv("ADMIN_INITIAL_FULL_NAME", "Sistem Yoneticisi").strip()
    password = os.getenv("ADMIN_INITIAL_PASSWORD", "")
    admin = session.scalar(select(User).where(User.role == "admin").order_by(User.created_at))
    if admin is None:
        if len(password) < 10:
            raise RuntimeError("ADMIN_INITIAL_PASSWORD must contain at least 10 characters.")
        admin = User(
            id=uuid4(), username=username, email=email, full_name=full_name,
            password_hash=hash_password(password), role="admin", is_active=True,
        )
        session.add(admin)
        session.flush()
    session.execute(update(Person).where(Person.owner_user_id.is_(None)).values(owner_user_id=admin.id, is_global=True))
    session.execute(update(AnonymousIdentity).where(AnonymousIdentity.owner_user_id.is_(None)).values(owner_user_id=admin.id))
    session.execute(update(RecognitionProcess).where(RecognitionProcess.owner_user_id.is_(None)).values(owner_user_id=admin.id))
    session.commit()
    return admin


def user_by_identifier(session: Session, identifier: str) -> Optional[User]:
    normalized = identifier.strip().lower()
    return session.scalar(select(User).where(or_(User.username == normalized, User.email == normalized)))
