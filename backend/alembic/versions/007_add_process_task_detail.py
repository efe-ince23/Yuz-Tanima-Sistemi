"""Add explicit task details to recognition process logs.

Revision ID: 007
Revises: 006
Create Date: 2026-08-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "recognition_processes",
        sa.Column("task_detail", sa.JSON(), nullable=True),
    )
    op.execute(
        """
        UPDATE recognition_processes AS process
        SET task_detail = json_build_object(
            'operation_type', process.operation_type,
            'processed_face_count', process.face_count,
            'faces', COALESCE(
                (
                    SELECT json_agg(
                        json_build_object(
                            'face_id', event.face_id,
                            'status', event.face_status
                        )
                        ORDER BY event.id
                    )
                    FROM recognition_events AS event
                    WHERE event.process_id = process.process_id
                ),
                '[]'::json
            ),
            'status', process.status
        )
        WHERE process.task_detail IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("recognition_processes", "task_detail")
