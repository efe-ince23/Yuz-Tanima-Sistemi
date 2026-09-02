"""Backfill process IDs for recognition events created before process tracking.

Revision ID: 009
Revises: 008
Create Date: 2026-08-20
"""
from typing import Sequence, Union

from alembic import op


revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        DECLARE
            event_row RECORD;
            generated_process_id UUID;
            generated_status TEXT;
        BEGIN
            FOR event_row IN
                SELECT id, face_id, face_status, recognized, created_at
                FROM recognition_events
                WHERE process_id IS NULL AND face_id IS NOT NULL
                ORDER BY id
            LOOP
                generated_process_id := gen_random_uuid();
                generated_status := CASE
                    WHEN event_row.recognized THEN 'recognized'
                    ELSE 'unrecognized'
                END;

                INSERT INTO recognition_processes (
                    process_id,
                    operation_type,
                    status,
                    http_status,
                    face_count,
                    task_detail,
                    result,
                    error_detail,
                    created_at,
                    completed_at
                ) VALUES (
                    generated_process_id,
                    'identify',
                    generated_status,
                    200,
                    1,
                    json_build_object(
                        'operation_type', 'identify',
                        'processed_face_count', 1,
                        'faces', json_build_array(
                            json_build_object(
                                'face_id', event_row.face_id,
                                'status', event_row.face_status
                            )
                        ),
                        'status', generated_status,
                        'legacy_backfill', true
                    ),
                    NULL,
                    NULL,
                    event_row.created_at,
                    event_row.created_at
                );

                UPDATE recognition_events
                SET process_id = generated_process_id
                WHERE id = event_row.id;
            END LOOP;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE recognition_events AS event
        SET process_id = NULL
        FROM recognition_processes AS process
        WHERE event.process_id = process.process_id
          AND process.task_detail ->> 'legacy_backfill' = 'true'
        """
    )
    op.execute(
        """
        DELETE FROM recognition_processes
        WHERE task_detail ->> 'legacy_backfill' = 'true'
        """
    )
