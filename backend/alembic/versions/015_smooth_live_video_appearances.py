"""Smooth live-camera appearance ranges across one missed sample.

Revision ID: 015
Revises: 014
Create Date: 2026-09-04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text(
            """
            DELETE FROM video_appearance_segments AS segment
            USING recognition_processes AS process
            WHERE segment.process_id = process.process_id
              AND process.task_detail ->> 'source_type' = 'live_camera'
            """
        )
    )
    connection.execute(
        sa.text(
            """
            WITH live_observations AS (
                SELECT
                    observation.*,
                    job.duration_seconds,
                    job.sampled_frame_count,
                    job.source_fps,
                    GREATEST(
                        1,
                        ROUND(
                            COALESCE(job.duration_seconds, 0) * 1000
                            / GREATEST(job.sampled_frame_count, 1)
                        )::bigint
                    ) AS sample_interval_ms,
                    LAG(observation.timestamp_ms) OVER (
                        PARTITION BY observation.track_id
                        ORDER BY observation.timestamp_ms, observation.id
                    ) AS previous_timestamp_ms
                FROM video_face_observations AS observation
                JOIN video_jobs AS job
                  ON job.process_id = observation.process_id
                JOIN recognition_processes AS process
                  ON process.process_id = observation.process_id
                WHERE process.task_detail ->> 'source_type' = 'live_camera'
            ),
            marked AS (
                SELECT
                    live_observations.*,
                    LEAST(
                        2500,
                        GREATEST(750, ROUND(sample_interval_ms * 2.4)::bigint)
                    ) AS adaptive_gap_ms,
                    CASE
                        WHEN previous_timestamp_ms IS NULL
                          OR timestamp_ms - previous_timestamp_ms > LEAST(
                              2500,
                              GREATEST(750, ROUND(sample_interval_ms * 2.4)::bigint)
                          )
                        THEN 1 ELSE 0
                    END AS starts_group
                FROM live_observations
            ),
            grouped AS (
                SELECT
                    marked.*,
                    SUM(starts_group) OVER (
                        PARTITION BY track_id
                        ORDER BY timestamp_ms, id
                    ) AS group_number
                FROM marked
            )
            INSERT INTO video_appearance_segments (
                process_id,
                track_id,
                face_id,
                face_status,
                start_ms,
                end_ms,
                start_frame,
                end_frame,
                observation_count,
                max_recognition_confidence,
                average_recognition_confidence
            )
            SELECT
                process_id,
                track_id,
                face_id,
                MAX(face_status),
                GREATEST(
                    0,
                    MIN(timestamp_ms)
                    - LEAST(
                        MAX(adaptive_gap_ms) / 2,
                        GREATEST(100, MAX(sample_interval_ms) / 2)
                    )
                ),
                LEAST(
                    ROUND(COALESCE(MAX(duration_seconds), 0) * 1000)::bigint,
                    MAX(timestamp_ms)
                    + LEAST(
                        MAX(adaptive_gap_ms) / 2,
                        GREATEST(100, MAX(sample_interval_ms) / 2)
                    )
                ),
                GREATEST(
                    0,
                    ROUND(
                        GREATEST(
                            0,
                            MIN(timestamp_ms)
                            - LEAST(
                                MAX(adaptive_gap_ms) / 2,
                                GREATEST(100, MAX(sample_interval_ms) / 2)
                            )
                        ) * COALESCE(MAX(source_fps), 25.0) / 1000
                    )::integer
                ),
                GREATEST(
                    0,
                    ROUND(
                        LEAST(
                            ROUND(COALESCE(MAX(duration_seconds), 0) * 1000)::bigint,
                            MAX(timestamp_ms)
                            + LEAST(
                                MAX(adaptive_gap_ms) / 2,
                                GREATEST(100, MAX(sample_interval_ms) / 2)
                            )
                        ) * COALESCE(MAX(source_fps), 25.0) / 1000
                    )::integer
                ),
                COUNT(*)::integer,
                MAX(recognition_confidence),
                AVG(recognition_confidence)
            FROM grouped
            GROUP BY process_id, track_id, face_id, group_number
            """
        )
    )


def downgrade() -> None:
    # Appearance rows are derived and can be regenerated from observations.
    pass
