"""Add source image metadata for photo recognition history.

Revision ID: 013
Revises: 012
Create Date: 2026-08-28
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("recognition_processes", sa.Column("source_image_path", sa.String(length=500), nullable=True))
    op.add_column("recognition_processes", sa.Column("source_filename", sa.String(length=255), nullable=True))
    op.add_column("recognition_processes", sa.Column("source_content_type", sa.String(length=100), nullable=True))
    op.add_column("recognition_processes", sa.Column("source_file_size_bytes", sa.BigInteger(), nullable=True))
    op.add_column("recognition_processes", sa.Column("source_image_width", sa.Integer(), nullable=True))
    op.add_column("recognition_processes", sa.Column("source_image_height", sa.Integer(), nullable=True))
    op.create_index(
        "ix_recognition_processes_owner_photo_created",
        "recognition_processes",
        ["owner_user_id", "operation_type", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_recognition_processes_owner_photo_created", table_name="recognition_processes")
    op.drop_column("recognition_processes", "source_image_height")
    op.drop_column("recognition_processes", "source_image_width")
    op.drop_column("recognition_processes", "source_file_size_bytes")
    op.drop_column("recognition_processes", "source_content_type")
    op.drop_column("recognition_processes", "source_filename")
    op.drop_column("recognition_processes", "source_image_path")
