"""Add an index for face history queries.

Revision ID: 008
Revises: 007
Create Date: 2026-08-20
"""
from typing import Sequence, Union

from alembic import op


revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_recognition_events_face_id_created_at",
        "recognition_events",
        ["face_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_recognition_events_face_id_created_at",
        table_name="recognition_events",
    )
