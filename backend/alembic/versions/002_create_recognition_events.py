"""Create recognition events table.

Revision ID: 002
Revises: 001
Create Date: 2026-08-18
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "recognition_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("recognized", sa.Boolean(), nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=True),
        sa.Column("similarity", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["person_id"], ["persons.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_recognition_events_created_at",
        "recognition_events",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_recognition_events_person_id",
        "recognition_events",
        ["person_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_recognition_events_person_id", table_name="recognition_events")
    op.drop_index("ix_recognition_events_created_at", table_name="recognition_events")
    op.drop_table("recognition_events")
