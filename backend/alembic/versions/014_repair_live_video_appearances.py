"""Repair sparse live-camera appearance ranges.

Revision ID: 014
Revises: 013
Create Date: 2026-09-04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "014"
down_revision: Union[str, None] = "013"
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
                        GREATEST(750, ROUND(sample_interval_ms * 1.8)::bigint)
                    ) AS adaptive_gap_ms,
                    CASE
                        WHEN previous_timestamp_ms IS NULL
                          OR timestamp_ms - previous_timestamp_ms > LEAST(
                              2500,
                              GREATEST(750, ROUND(sample_interval_ms * 1.8)::bigint)
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
                grouped.process_id,
                grouped.track_id,
                grouped.face_id,
                MAX(grouped.face_status),
                GREATEST(
                    0,
                    MIN(grouped.timestamp_ms)
                    - LEAST(
                        MAX(grouped.adaptive_gap_ms) / 2,
                        GREATEST(100, MAX(grouped.sample_interval_ms) / 2)
                    )
                ),
                LEAST(
                    ROUND(COALESCE(MAX(grouped.duration_seconds), 0) * 1000)::bigint,
                    MAX(grouped.timestamp_ms)
                    + LEAST(
                        MAX(grouped.adaptive_gap_ms) / 2,
                        GREATEST(100, MAX(grouped.sample_interval_ms) / 2)
                    )
                ),
                GREATEST(
                    0,
                    ROUND(
                        GREATEST(
                            0,
                            MIN(grouped.timestamp_ms)
                            - LEAST(
                                MAX(grouped.adaptive_gap_ms) / 2,
                                GREATEST(100, MAX(grouped.sample_interval_ms) / 2)
                            )
                        ) * MAX(job.source_fps) / 1000
                    )::integer
                ),
                GREATEST(
                    0,
                    ROUND(
                        LEAST(
                            ROUND(COALESCE(MAX(grouped.duration_seconds), 0) * 1000)::bigint,
                            MAX(grouped.timestamp_ms)
                            + LEAST(
                                MAX(grouped.adaptive_gap_ms) / 2,
                                GREATEST(100, MAX(grouped.sample_interval_ms) / 2)
                            )
                        ) * MAX(job.source_fps) / 1000
                    )::integer
                ),
                COUNT(*)::integer,
                MAX(grouped.recognition_confidence),
                AVG(grouped.recognition_confidence)
            FROM grouped
            JOIN video_jobs AS job ON job.process_id = grouped.process_id
            GROUP BY
                grouped.process_id,
                grouped.track_id,
                grouped.face_id,
                grouped.group_number
            """
        )
    )


def downgrade() -> None:
    # Appearance rows are derived data and will be regenerated on the next analysis.
    pass
