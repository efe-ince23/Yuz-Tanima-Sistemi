from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api_errors import api_error
from app.auth import get_current_user
from app.database import get_database_session
from app.face_storage import image_url
from app.models import (
    Person,
    RecognitionEvent,
    RecognitionProcess,
    User,
    VideoJob,
    VideoTrack,
)
from app.schemas import (
    AppearanceSearchIntervalResponse,
    AppearanceSearchItemResponse,
    AppearanceSearchResponse,
)


router = APIRouter(prefix="/api/search", tags=["search"])


def _resolve_owner_scope(user: User, requested_owner_id: Optional[UUID]) -> UUID:
    if requested_owner_id is not None and requested_owner_id != user.id:
        raise api_error(
            403,
            "SEARCH_SCOPE_FORBIDDEN",
            "Baska bir kullanicinin kayitlari aranamaz.",
        )
    return user.id


def _validate_ranges(
    date_from: Optional[datetime],
    date_to: Optional[datetime],
    min_confidence: Optional[float],
    max_confidence: Optional[float],
) -> None:
    if date_from is not None and date_to is not None and date_from > date_to:
        raise api_error(422, "INVALID_DATE_RANGE", "Baslangic tarihi bitis tarihinden sonra olamaz.")
    if (
        min_confidence is not None
        and max_confidence is not None
        and min_confidence > max_confidence
    ):
        raise api_error(422, "INVALID_CONFIDENCE_RANGE", "Minimum guven skoru maksimumdan buyuk olamaz.")


def _matches_filters(
    item: AppearanceSearchItemResponse,
    query: Optional[str],
    face_id: Optional[UUID],
    identity_status: Optional[str],
    min_confidence: Optional[float],
    max_confidence: Optional[float],
) -> bool:
    if face_id is not None and item.face_id != face_id:
        return False
    if identity_status == "known" and item.status != "known":
        return False
    if identity_status == "anonymous" and item.status == "known":
        return False
    if min_confidence is not None and (
        item.confidence is None or item.confidence < min_confidence
    ):
        return False
    if max_confidence is not None and (
        item.confidence is None or item.confidence > max_confidence
    ):
        return False
    if query:
        needle = query.casefold().strip()
        haystack = " ".join(
            value
            for value in (
                str(item.face_id),
                item.first_name,
                item.last_name,
            )
            if value
        ).casefold()
        if needle not in haystack:
            return False
    return True


def _photo_items(
    session: Session,
    owner_id: Optional[UUID],
    date_from: Optional[datetime],
    date_to: Optional[datetime],
) -> list[AppearanceSearchItemResponse]:
    statement = (
        select(RecognitionEvent, RecognitionProcess, User, Person)
        .join(RecognitionProcess, RecognitionProcess.process_id == RecognitionEvent.process_id)
        .outerjoin(User, User.id == RecognitionProcess.owner_user_id)
        .outerjoin(Person, Person.id == RecognitionEvent.person_id)
        .where(
            RecognitionProcess.operation_type == "identify",
            RecognitionProcess.source_image_path.is_not(None),
            RecognitionEvent.face_id.is_not(None),
        )
    )
    if owner_id is not None:
        statement = statement.where(RecognitionProcess.owner_user_id == owner_id)
    if date_from is not None:
        statement = statement.where(RecognitionProcess.created_at >= date_from)
    if date_to is not None:
        statement = statement.where(RecognitionProcess.created_at <= date_to)

    grouped: dict[tuple[UUID, UUID], AppearanceSearchItemResponse] = {}
    for event, process, owner, person in session.execute(statement).all():
        key = (process.process_id, event.face_id)
        existing = grouped.get(key)
        if existing is not None:
            existing.observation_count += 1
            if event.similarity is not None and (
                existing.confidence is None or event.similarity > existing.confidence
            ):
                existing.confidence = event.similarity
            continue
        grouped[key] = AppearanceSearchItemResponse(
            source_type="photo",
            process_id=process.process_id,
            face_id=event.face_id,
            status=event.face_status or ("known" if event.recognized else "anonymous"),
            person_id=person.id if person else None,
            first_name=person.first_name if person else None,
            last_name=person.last_name if person else None,
            metadata={"description": person.description} if person else None,
            owner_user_id=process.owner_user_id,
            owner_username=owner.username if owner else None,
            owner_full_name=owner.full_name if owner else None,
            occurred_at=process.created_at,
            confidence=event.similarity,
            original_filename=process.source_filename,
            preview_url=f"/api/photos/{process.process_id}/content",
            content_url=f"/api/photos/{process.process_id}/content",
            observation_count=1,
            first_seen_ms=None,
            last_seen_ms=None,
            intervals=[],
        )
    return list(grouped.values())


def _video_items(
    session: Session,
    owner_id: Optional[UUID],
    date_from: Optional[datetime],
    date_to: Optional[datetime],
) -> list[AppearanceSearchItemResponse]:
    statement = (
        select(VideoTrack, VideoJob, RecognitionProcess, User, Person)
        .join(VideoJob, VideoJob.process_id == VideoTrack.process_id)
        .join(RecognitionProcess, RecognitionProcess.process_id == VideoJob.process_id)
        .outerjoin(User, User.id == RecognitionProcess.owner_user_id)
        .outerjoin(Person, Person.face_id == VideoTrack.face_id)
        .options(selectinload(VideoTrack.appearance_segments))
        .where(VideoTrack.face_id.is_not(None), VideoJob.status == "completed")
    )
    if owner_id is not None:
        statement = statement.where(RecognitionProcess.owner_user_id == owner_id)
    if date_from is not None:
        statement = statement.where(VideoJob.created_at >= date_from)
    if date_to is not None:
        statement = statement.where(VideoJob.created_at <= date_to)

    rows: dict[tuple[UUID, UUID], dict[str, object]] = defaultdict(dict)
    for track, job, process, owner, person in session.execute(statement).all():
        key = (track.process_id, track.face_id)
        bucket = rows[key]
        if not bucket:
            bucket.update(
                track=track,
                job=job,
                process=process,
                owner=owner,
                person=person,
                observation_count=0,
                first_seen_ms=track.first_seen_ms,
                last_seen_ms=track.last_seen_ms,
                confidence=None,
                preview_url=None,
                intervals=[],
            )
        bucket["observation_count"] = int(bucket["observation_count"]) + track.observation_count
        bucket["first_seen_ms"] = min(int(bucket["first_seen_ms"]), track.first_seen_ms)
        bucket["last_seen_ms"] = max(int(bucket["last_seen_ms"]), track.last_seen_ms)
        confidence = track.best_recognition_confidence
        if confidence is not None and (
            bucket["confidence"] is None or confidence > float(bucket["confidence"])
        ):
            bucket["confidence"] = confidence
        if not bucket["preview_url"] and track.best_image_path:
            bucket["preview_url"] = image_url(track.best_image_path)
        bucket["intervals"].extend(
            AppearanceSearchIntervalResponse(start_ms=segment.start_ms, end_ms=segment.end_ms)
            for segment in track.appearance_segments
        )

    items = []
    for bucket in rows.values():
        track = bucket["track"]
        job = bucket["job"]
        process = bucket["process"]
        owner = bucket["owner"]
        person = bucket["person"]
        intervals = sorted(bucket["intervals"], key=lambda interval: interval.start_ms)
        items.append(
            AppearanceSearchItemResponse(
                source_type="video",
                process_id=process.process_id,
                face_id=track.face_id,
                status=track.face_status or ("known" if person else "anonymous"),
                person_id=person.id if person else None,
                first_name=person.first_name if person else None,
                last_name=person.last_name if person else None,
                metadata={"description": person.description} if person else None,
                owner_user_id=process.owner_user_id,
                owner_username=owner.username if owner else None,
                owner_full_name=owner.full_name if owner else None,
                occurred_at=job.created_at,
                confidence=bucket["confidence"],
                original_filename=job.original_filename,
                preview_url=bucket["preview_url"],
                content_url=f"/api/videos/{process.process_id}/content",
                observation_count=bucket["observation_count"],
                first_seen_ms=bucket["first_seen_ms"],
                last_seen_ms=bucket["last_seen_ms"],
                intervals=intervals,
            )
        )
    return items


@router.get("/appearances", response_model=AppearanceSearchResponse, include_in_schema=False)
def search_appearances(
    q: Optional[str] = Query(None, max_length=150),
    face_id: Optional[UUID] = None,
    identity_status: Optional[Literal["known", "anonymous"]] = None,
    source_type: Literal["all", "photo", "video"] = "all",
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_confidence: Optional[float] = Query(None, ge=-1.0, le=1.0),
    max_confidence: Optional[float] = Query(None, ge=-1.0, le=1.0),
    owner_user_id: Optional[UUID] = None,
    sort: Literal["newest", "oldest", "confidence"] = "newest",
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> AppearanceSearchResponse:
    _validate_ranges(date_from, date_to, min_confidence, max_confidence)
    effective_owner_id = _resolve_owner_scope(user, owner_user_id)

    items: list[AppearanceSearchItemResponse] = []
    if source_type in {"all", "photo"}:
        items.extend(_photo_items(session, effective_owner_id, date_from, date_to))
    if source_type in {"all", "video"}:
        items.extend(_video_items(session, effective_owner_id, date_from, date_to))
    items = [
        item
        for item in items
        if _matches_filters(
            item, q, face_id, identity_status, min_confidence, max_confidence
        )
    ]
    if sort == "oldest":
        items.sort(key=lambda item: (item.occurred_at, str(item.process_id)))
    elif sort == "confidence":
        items.sort(
            key=lambda item: (item.confidence is not None, item.confidence or -2.0, item.occurred_at),
            reverse=True,
        )
    else:
        items.sort(key=lambda item: (item.occurred_at, str(item.process_id)), reverse=True)
    return AppearanceSearchResponse(
        total=len(items),
        limit=limit,
        offset=offset,
        items=items[offset : offset + limit],
    )
