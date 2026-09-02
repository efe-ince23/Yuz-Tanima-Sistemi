from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'user')", name="ck_users_role"),
        Index("uq_users_username_lower", text("lower(username)"), unique=True),
        Index("uq_users_email_lower", text("lower(email)"), unique=True),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid4
    )
    username: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(150), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    sessions: Mapped[List["AuthSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    __table_args__ = (
        Index("ix_auth_sessions_user_id", "user_id"),
        Index("ix_auth_sessions_expires_at", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid4
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    refresh_token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    user: Mapped[User] = relationship(back_populates="sessions")


class Person(Base):
    __tablename__ = "persons"
    __table_args__ = (
        Index("uq_persons_source_external_id", "source", "external_id", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    face_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), default=uuid4, unique=True, nullable=False
    )
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    owner_user_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    is_global: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    face_images: Mapped[List["FaceImage"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    enrolled_anonymous_identity: Mapped[Optional["AnonymousIdentity"]] = relationship(
        back_populates="person",
        uselist=False,
    )


class FaceImage(Base):
    __tablename__ = "face_images"
    __table_args__ = (Index("ix_face_images_person_id", "person_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=False
    )
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(512), nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    person: Mapped[Person] = relationship(back_populates="face_images")


class AnonymousIdentity(Base):
    __tablename__ = "anonymous_identities"

    id: Mapped[int] = mapped_column(primary_key=True)
    face_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), default=uuid4, unique=True, nullable=False
    )
    person_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"), unique=True, nullable=True
    )
    owner_user_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    observation_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    embeddings: Mapped[List["AnonymousFaceEmbedding"]] = relationship(
        back_populates="identity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    person: Mapped[Optional[Person]] = relationship(
        back_populates="enrolled_anonymous_identity"
    )


class AnonymousFaceEmbedding(Base):
    __tablename__ = "anonymous_face_embeddings"
    __table_args__ = (
        Index("ix_anonymous_face_embeddings_identity_id", "anonymous_identity_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    anonymous_identity_id: Mapped[int] = mapped_column(
        ForeignKey("anonymous_identities.id", ondelete="CASCADE"), nullable=False
    )
    embedding: Mapped[List[float]] = mapped_column(Vector(512), nullable=False)
    image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    identity: Mapped[AnonymousIdentity] = relationship(back_populates="embeddings")


class RecognitionProcess(Base):
    __tablename__ = "recognition_processes"
    __table_args__ = (Index("ix_recognition_processes_created_at", "created_at"),)

    process_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid4
    )
    operation_type: Mapped[str] = mapped_column(String(30), nullable=False)
    owner_user_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(30), default="processing", nullable=False)
    http_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    face_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    task_detail: Mapped[Optional[Dict[str, object]]] = mapped_column(JSON, nullable=True)
    result: Mapped[Optional[Dict[str, object]]] = mapped_column(JSON, nullable=True)
    error_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_content_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    source_image_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_image_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    events: Mapped[List["RecognitionEvent"]] = relationship(
        back_populates="process"
    )
    video_job: Mapped[Optional["VideoJob"]] = relationship(
        back_populates="process",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class VideoJob(Base):
    __tablename__ = "video_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')",
            name="ck_video_jobs_status",
        ),
        CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_video_jobs_progress_percent",
        ),
        Index("ix_video_jobs_status_created_at", "status", "created_at"),
    )

    process_id: Mapped[UUID] = mapped_column(
        ForeignKey("recognition_processes.process_id", ondelete="CASCADE"),
        primary_key=True,
    )
    status: Mapped[str] = mapped_column(String(20), default="queued", nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    object_path: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    frame_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sampled_frame_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_frame_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    detected_face_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    unique_face_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    error_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    process: Mapped[RecognitionProcess] = relationship(back_populates="video_job")
    tracks: Mapped[List["VideoTrack"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    observations: Mapped[List["VideoFaceObservation"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    appearance_segments: Mapped[List["VideoAppearanceSegment"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class VideoTrack(Base):
    __tablename__ = "video_tracks"
    __table_args__ = (
        CheckConstraint(
            "face_status IS NULL OR face_status IN "
            "('known', 'anonymous', 'new_anonymous')",
            name="ck_video_tracks_face_status",
        ),
        UniqueConstraint(
            "process_id", "track_number", name="uq_video_tracks_process_track_number"
        ),
        Index("ix_video_tracks_face_id", "face_id"),
        Index("ix_video_tracks_process_id", "process_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    process_id: Mapped[UUID] = mapped_column(
        ForeignKey("video_jobs.process_id", ondelete="CASCADE"), nullable=False
    )
    track_number: Mapped[int] = mapped_column(Integer, nullable=False)
    face_id: Mapped[Optional[UUID]] = mapped_column(Uuid(as_uuid=True), nullable=True)
    face_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    first_seen_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_seen_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    best_detection_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    best_recognition_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    best_frame_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    best_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    job: Mapped[VideoJob] = relationship(back_populates="tracks")
    observations: Mapped[List["VideoFaceObservation"]] = relationship(
        back_populates="track",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    appearance_segments: Mapped[List["VideoAppearanceSegment"]] = relationship(
        back_populates="track",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class VideoFaceObservation(Base):
    __tablename__ = "video_face_observations"
    __table_args__ = (
        CheckConstraint(
            "face_status IS NULL OR face_status IN "
            "('known', 'anonymous', 'new_anonymous')",
            name="ck_video_face_observations_status",
        ),
        CheckConstraint(
            "bbox_x1 >= 0 AND bbox_x1 <= 1 AND "
            "bbox_y1 >= 0 AND bbox_y1 <= 1 AND "
            "bbox_x2 >= 0 AND bbox_x2 <= 1 AND "
            "bbox_y2 >= 0 AND bbox_y2 <= 1 AND "
            "bbox_x2 >= bbox_x1 AND bbox_y2 >= bbox_y1",
            name="ck_video_face_observations_bbox",
        ),
        UniqueConstraint(
            "track_id",
            "frame_number",
            name="uq_video_face_observations_track_frame",
        ),
        Index(
            "ix_video_face_observations_process_timestamp",
            "process_id",
            "timestamp_ms",
        ),
        Index(
            "ix_video_face_observations_face_timestamp",
            "face_id",
            "timestamp_ms",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    process_id: Mapped[UUID] = mapped_column(
        ForeignKey("video_jobs.process_id", ondelete="CASCADE"), nullable=False
    )
    track_id: Mapped[int] = mapped_column(
        ForeignKey("video_tracks.id", ondelete="CASCADE"), nullable=False
    )
    face_id: Mapped[Optional[UUID]] = mapped_column(Uuid(as_uuid=True), nullable=True)
    face_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    recognition_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[VideoJob] = relationship(back_populates="observations")
    track: Mapped[VideoTrack] = relationship(back_populates="observations")


class VideoAppearanceSegment(Base):
    __tablename__ = "video_appearance_segments"
    __table_args__ = (
        CheckConstraint(
            "face_status IN ('known', 'anonymous', 'new_anonymous')",
            name="ck_video_appearance_segments_status",
        ),
        CheckConstraint(
            "end_ms >= start_ms AND end_frame >= start_frame",
            name="ck_video_appearance_segments_range",
        ),
        Index(
            "ix_video_appearance_segments_process_start",
            "process_id",
            "start_ms",
        ),
        Index(
            "ix_video_appearance_segments_face_start", "face_id", "start_ms"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    process_id: Mapped[UUID] = mapped_column(
        ForeignKey("video_jobs.process_id", ondelete="CASCADE"), nullable=False
    )
    track_id: Mapped[int] = mapped_column(
        ForeignKey("video_tracks.id", ondelete="CASCADE"), nullable=False
    )
    face_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    face_status: Mapped[str] = mapped_column(String(20), nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_recognition_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    average_recognition_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[VideoJob] = relationship(back_populates="appearance_segments")
    track: Mapped[VideoTrack] = relationship(back_populates="appearance_segments")


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"
    __table_args__ = (
        Index("ix_recognition_events_created_at", "created_at"),
        Index("ix_recognition_events_person_id", "person_id"),
        Index("ix_recognition_events_face_id", "face_id"),
        Index("ix_recognition_events_face_id_created_at", "face_id", "created_at"),
        Index("ix_recognition_events_process_id", "process_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    process_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("recognition_processes.process_id", ondelete="SET NULL"),
        nullable=True,
    )
    recognized: Mapped[bool] = mapped_column(Boolean, nullable=False)
    person_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"), nullable=True
    )
    face_id: Mapped[Optional[UUID]] = mapped_column(Uuid(as_uuid=True), nullable=True)
    face_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    process: Mapped[Optional[RecognitionProcess]] = relationship(
        back_populates="events"
    )


class DatasetImportItem(Base):
    __tablename__ = "dataset_import_items"
    __table_args__ = (
        Index("ix_dataset_import_items_dataset_status", "dataset_name", "status"),
        Index(
            "uq_dataset_import_items_dataset_source_path",
            "dataset_name",
            "source_path",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(50), nullable=False)
    external_identity: Mapped[str] = mapped_column(String(255), nullable=False)
    source_path: Mapped[str] = mapped_column(String(500), nullable=False)
    person_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=True
    )
    face_image_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("face_images.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    error_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
