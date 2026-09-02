from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, List, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api_errors import api_error
from app.auth import (
    ACCESS_COOKIE,
    ACCESS_MINUTES,
    COOKIE_SECURE,
    REFRESH_COOKIE,
    REFRESH_DAYS,
    decode_access_token,
    encode_access_token,
    get_current_user,
    hash_password,
    new_refresh_token,
    refresh_token_hash,
    require_admin,
    user_by_identifier,
    verify_password,
)
from app.database import get_database_session
from app.models import AuthSession, User
from app.schemas import AdminUserUpdate, AuthResponse, LoginRequest, RegisterRequest, UserResponse


router = APIRouter(prefix="/api/auth", tags=["authentication"])
admin_router = APIRouter(prefix="/api/admin/users", tags=["admin users"])
_attempt_lock = Lock()
_failed_attempts: Dict[str, Tuple[int, datetime]] = {}


def _client_key(request: Request, identifier: str) -> str:
    address = request.client.host if request.client else "unknown"
    return f"{address}:{identifier.strip().lower()}"


def _check_rate_limit(key: str) -> None:
    now = datetime.now(timezone.utc)
    with _attempt_lock:
        count, started = _failed_attempts.get(key, (0, now))
        if now - started > timedelta(minutes=15):
            _failed_attempts.pop(key, None)
            return
        if count >= 5:
            raise api_error(429, "AUTH_RATE_LIMITED", "Cok fazla basarisiz giris denemesi. Daha sonra tekrar deneyin.")


def _failed_login(key: str) -> None:
    now = datetime.now(timezone.utc)
    with _attempt_lock:
        count, started = _failed_attempts.get(key, (0, now))
        if now - started > timedelta(minutes=15):
            count, started = 0, now
        _failed_attempts[key] = (count + 1, started)


def _set_auth_cookies(response: Response, user: User, auth_session: AuthSession, refresh_token: str) -> datetime:
    access_token, access_expires_at = encode_access_token(user, auth_session.id)
    response.set_cookie(ACCESS_COOKIE, access_token, max_age=ACCESS_MINUTES * 60, httponly=True, secure=COOKIE_SECURE, samesite="strict", path="/")
    response.set_cookie(REFRESH_COOKIE, refresh_token, max_age=REFRESH_DAYS * 86400, httponly=True, secure=COOKIE_SECURE, samesite="strict", path="/api/auth")
    return access_expires_at


def _create_session(session: Session, request: Request, user: User) -> Tuple[AuthSession, str]:
    refresh_token = new_refresh_token()
    auth_session = AuthSession(
        user_id=user.id,
        refresh_token_hash=refresh_token_hash(refresh_token),
        expires_at=datetime.now(timezone.utc) + timedelta(days=REFRESH_DAYS),
        user_agent=(request.headers.get("user-agent") or "")[:500] or None,
        ip_address=(request.client.host if request.client else None),
    )
    session.add(auth_session)
    session.flush()
    return auth_session, refresh_token


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
def register(payload: RegisterRequest, request: Request, response: Response, session: Session = Depends(get_database_session)) -> AuthResponse:
    exists = session.scalar(select(User.id).where(or_(func.lower(User.username) == payload.username, func.lower(User.email) == payload.email)))
    if exists is not None:
        raise api_error(409, "USER_EXISTS", "Kullanici adi veya e-posta zaten kayitli.")
    user = User(username=payload.username, email=payload.email, full_name=payload.full_name, password_hash=hash_password(payload.password), role="user", is_active=True, last_login_at=datetime.now(timezone.utc))
    session.add(user)
    try:
        session.flush()
        auth_session, refresh_token = _create_session(session, request, user)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise api_error(409, "USER_EXISTS", "Kullanici adi veya e-posta zaten kayitli.") from error
    expires_at = _set_auth_cookies(response, user, auth_session, refresh_token)
    return AuthResponse(user=UserResponse.model_validate(user), access_expires_at=expires_at)


@router.post("/login", response_model=AuthResponse, include_in_schema=False)
def login(payload: LoginRequest, request: Request, response: Response, session: Session = Depends(get_database_session)) -> AuthResponse:
    key = _client_key(request, payload.identifier)
    _check_rate_limit(key)
    user = user_by_identifier(session, payload.identifier)
    if user is None or not verify_password(user.password_hash, payload.password):
        _failed_login(key)
        raise api_error(401, "AUTH_FAILED", "Kullanici adi/e-posta veya parola hatali.")
    if not user.is_active:
        raise api_error(403, "AUTH_INACTIVE", "Kullanici hesabi yonetici tarafindan pasiflestirilmis.")
    with _attempt_lock:
        _failed_attempts.pop(key, None)
    user.last_login_at = datetime.now(timezone.utc)
    auth_session, refresh_token = _create_session(session, request, user)
    session.commit()
    expires_at = _set_auth_cookies(response, user, auth_session, refresh_token)
    return AuthResponse(user=UserResponse.model_validate(user), access_expires_at=expires_at)


@router.post("/refresh", response_model=AuthResponse, include_in_schema=False)
def refresh(request: Request, response: Response, session: Session = Depends(get_database_session)) -> AuthResponse:
    old_token = request.cookies.get(REFRESH_COOKIE)
    if not old_token:
        raise api_error(401, "REFRESH_REQUIRED", "Oturum yenileme bilgisi bulunamadi.")
    auth_session = session.scalar(select(AuthSession).where(AuthSession.refresh_token_hash == refresh_token_hash(old_token)).with_for_update())
    now = datetime.now(timezone.utc)
    if auth_session is None or auth_session.revoked_at is not None or auth_session.expires_at <= now:
        raise api_error(401, "REFRESH_INVALID", "Oturum yenileme bilgisi gecersiz.")
    user = session.get(User, auth_session.user_id)
    if user is None or not user.is_active:
        raise api_error(401, "AUTH_INACTIVE", "Kullanici hesabi aktif degil.")
    auth_session.revoked_at = now
    new_session, refresh_token = _create_session(session, request, user)
    session.commit()
    expires_at = _set_auth_cookies(response, user, new_session, refresh_token)
    return AuthResponse(user=UserResponse.model_validate(user), access_expires_at=expires_at)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, include_in_schema=False)
def logout(request: Request, response: Response, session: Session = Depends(get_database_session)) -> Response:
    token = request.cookies.get(REFRESH_COOKIE)
    if token:
        auth_session = session.scalar(select(AuthSession).where(AuthSession.refresh_token_hash == refresh_token_hash(token)))
        if auth_session is not None and auth_session.revoked_at is None:
            auth_session.revoked_at = datetime.now(timezone.utc)
            session.commit()
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path="/api/auth")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=UserResponse, include_in_schema=False)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(user)


@admin_router.get("", response_model=List[UserResponse], include_in_schema=False)
def list_users(_admin: User = Depends(require_admin), session: Session = Depends(get_database_session)) -> List[UserResponse]:
    users = session.scalars(select(User).order_by(User.created_at.desc())).all()
    return [UserResponse.model_validate(user) for user in users]


@admin_router.patch("/{user_id}", response_model=UserResponse, include_in_schema=False)
def update_user(user_id: UUID, payload: AdminUserUpdate, admin: User = Depends(require_admin), session: Session = Depends(get_database_session)) -> UserResponse:
    user = session.get(User, user_id)
    if user is None:
        raise api_error(404, "USER_NOT_FOUND", "Kullanici bulunamadi.")
    if user.id == admin.id and not payload.is_active:
        raise api_error(409, "ADMIN_SELF_DISABLE", "Kendi yonetici hesabinizi pasiflestiremezsiniz.")
    user.is_active = payload.is_active
    if not user.is_active:
        for auth_session in session.scalars(select(AuthSession).where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))):
            auth_session.revoked_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(user)
    return UserResponse.model_validate(user)
