"""Add video recognition storage.

Revision ID: 011
Revises: 010
Create Date: 2026-08-24
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "video_jobs",
        sa.Column("process_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("object_path", sa.String(length=500), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("source_fps", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("frame_count", sa.Integer(), nullable=True),
        sa.Column("sampled_frame_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("processed_frame_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("progress_percent", sa.Float(), server_default="0", nullable=False),
        sa.Column("detected_face_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("unique_face_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_code", sa.String(length=50), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')",
            name="ck_video_jobs_status",
        ),
        sa.CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_video_jobs_progress_percent",
        ),
        sa.ForeignKeyConstraint(
            ["process_id"],
            ["recognition_processes.process_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("process_id"),
    )
    op.create_index(
        "ix_video_jobs_status_created_at",
        "video_jobs",
        ["status", "created_at"],
    )

    op.create_table(
        "video_tracks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("process_id", sa.Uuid(), nullable=False),
        sa.Column("track_number", sa.Integer(), nullable=False),
        sa.Column("face_id", sa.Uuid(), nullable=True),
        sa.Column("face_status", sa.String(length=20), nullable=True),
        sa.Column("first_seen_ms", sa.BigInteger(), nullable=False),
        sa.Column("last_seen_ms", sa.BigInteger(), nullable=False),
        sa.Column("observation_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("best_detection_confidence", sa.Float(), nullable=True),
        sa.Column("best_recognition_confidence", sa.Float(), nullable=True),
        sa.Column("best_frame_number", sa.Integer(), nullable=True),
        sa.Column("best_image_path", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "face_status IS NULL OR face_status IN "
            "('known', 'anonymous', 'new_anonymous')",
            name="ck_video_tracks_face_status",
        ),
        sa.ForeignKeyConstraint(
            ["process_id"], ["video_jobs.process_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "process_id", "track_number", name="uq_video_tracks_process_track_number"
        ),
    )
    op.create_index("ix_video_tracks_face_id", "video_tracks", ["face_id"])
    op.create_index("ix_video_tracks_process_id", "video_tracks", ["process_id"])

    op.create_table(
        "video_face_observations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("process_id", sa.Uuid(), nullable=False),
        sa.Column("track_id", sa.BigInteger(), nullable=False),
        sa.Column("face_id", sa.Uuid(), nullable=True),
        sa.Column("face_status", sa.String(length=20), nullable=True),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("timestamp_ms", sa.BigInteger(), nullable=False),
        sa.Column("bbox_x1", sa.Float(), nullable=False),
        sa.Column("bbox_y1", sa.Float(), nullable=False),
        sa.Column("bbox_x2", sa.Float(), nullable=False),
        sa.Column("bbox_y2", sa.Float(), nullable=False),
        sa.Column("detection_confidence", sa.Float(), nullable=False),
        sa.Column("recognition_confidence", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "face_status IS NULL OR face_status IN "
            "('known', 'anonymous', 'new_anonymous')",
            name="ck_video_face_observations_status",
        ),
        sa.CheckConstraint(
            "bbox_x1 >= 0 AND bbox_x1 <= 1 AND "
            "bbox_y1 >= 0 AND bbox_y1 <= 1 AND "
            "bbox_x2 >= 0 AND bbox_x2 <= 1 AND "
            "bbox_y2 >= 0 AND bbox_y2 <= 1 AND "
            "bbox_x2 >= bbox_x1 AND bbox_y2 >= bbox_y1",
            name="ck_video_face_observations_bbox",
        ),
        sa.ForeignKeyConstraint(
            ["process_id"], ["video_jobs.process_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["track_id"], ["video_tracks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "track_id",
            "frame_number",
            name="uq_video_face_observations_track_frame",
        ),
    )
    op.create_index(
        "ix_video_face_observations_process_timestamp",
        "video_face_observations",
        ["process_id", "timestamp_ms"],
    )
    op.create_index(
        "ix_video_face_observations_face_timestamp",
        "video_face_observations",
        ["face_id", "timestamp_ms"],
    )

    op.create_table(
        "video_appearance_segments",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("process_id", sa.Uuid(), nullable=False),
        sa.Column("track_id", sa.BigInteger(), nullable=False),
        sa.Column("face_id", sa.Uuid(), nullable=False),
        sa.Column("face_status", sa.String(length=20), nullable=False),
        sa.Column("start_ms", sa.BigInteger(), nullable=False),
        sa.Column("end_ms", sa.BigInteger(), nullable=False),
        sa.Column("start_frame", sa.Integer(), nullable=False),
        sa.Column("end_frame", sa.Integer(), nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column("max_recognition_confidence", sa.Float(), nullable=True),
        sa.Column("average_recognition_confidence", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "face_status IN ('known', 'anonymous', 'new_anonymous')",
            name="ck_video_appearance_segments_status",
        ),
        sa.CheckConstraint(
            "end_ms >= start_ms AND end_frame >= start_frame",
            name="ck_video_appearance_segments_range",
        ),
        sa.ForeignKeyConstraint(
            ["process_id"], ["video_jobs.process_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["track_id"], ["video_tracks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_video_appearance_segments_process_start",
        "video_appearance_segments",
        ["process_id", "start_ms"],
    )
    op.create_index(
        "ix_video_appearance_segments_face_start",
        "video_appearance_segments",
        ["face_id", "start_ms"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_video_appearance_segments_face_start",
        table_name="video_appearance_segments",
    )
    op.drop_index(
        "ix_video_appearance_segments_process_start",
        table_name="video_appearance_segments",
    )
    op.drop_table("video_appearance_segments")
    op.drop_index(
        "ix_video_face_observations_face_timestamp",
        table_name="video_face_observations",
    )
    op.drop_index(
        "ix_video_face_observations_process_timestamp",
        table_name="video_face_observations",
    )
    op.drop_table("video_face_observations")
    op.drop_index("ix_video_tracks_process_id", table_name="video_tracks")
    op.drop_index("ix_video_tracks_face_id", table_name="video_tracks")
    op.drop_table("video_tracks")
    op.drop_index("ix_video_jobs_status_created_at", table_name="video_jobs")
    op.drop_table("video_jobs")
