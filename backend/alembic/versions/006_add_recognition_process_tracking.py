"""Add recognition process tracking.

Revision ID: 006
Revises: 005
Create Date: 2026-08-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "recognition_processes",
        sa.Column("process_id", sa.Uuid(), nullable=False),
        sa.Column("operation_type", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("face_count", sa.Integer(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("process_id"),
    )
    op.create_index(
        "ix_recognition_processes_created_at",
        "recognition_processes",
        ["created_at"],
    )
    op.add_column(
        "recognition_events",
        sa.Column("process_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_recognition_events_process_id",
        "recognition_events",
        "recognition_processes",
        ["process_id"],
        ["process_id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_recognition_events_process_id",
        "recognition_events",
        ["process_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_recognition_events_process_id", table_name="recognition_events")
    op.drop_constraint(
        "fk_recognition_events_process_id",
        "recognition_events",
        type_="foreignkey",
    )
    op.drop_column("recognition_events", "process_id")
    op.drop_index(
        "ix_recognition_processes_created_at",
        table_name="recognition_processes",
    )
    op.drop_table("recognition_processes")
