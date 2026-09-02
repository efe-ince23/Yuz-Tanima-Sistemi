"""Add persistent face IDs and anonymous identities.

Revision ID: 003
Revises: 002
Create Date: 2026-08-18
"""
from typing import Sequence, Union
import uuid

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "persons",
        sa.Column("face_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    connection = op.get_bind()
    person_ids = connection.execute(sa.text("SELECT id FROM persons")).scalars()
    for person_id in person_ids:
        connection.execute(
            sa.text("UPDATE persons SET face_id = :face_id WHERE id = :person_id"),
            {"face_id": uuid.uuid4(), "person_id": person_id},
        )
    op.alter_column("persons", "face_id", nullable=False)
    op.create_unique_constraint("uq_persons_face_id", "persons", ["face_id"])

    op.create_table(
        "anonymous_identities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("face_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("observation_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("face_id", name="uq_anonymous_identities_face_id"),
    )
    op.create_table(
        "anonymous_face_embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("anonymous_identity_id", sa.Integer(), nullable=False),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.Vector(dim=512),
            nullable=False,
        ),
        sa.Column("detection_confidence", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["anonymous_identity_id"],
            ["anonymous_identities.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_anonymous_face_embeddings_identity_id",
        "anonymous_face_embeddings",
        ["anonymous_identity_id"],
        unique=False,
    )

    op.add_column(
        "recognition_events",
        sa.Column("face_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "recognition_events",
        sa.Column("face_status", sa.String(length=20), nullable=True),
    )
    connection.execute(
        sa.text(
            """
            UPDATE recognition_events AS event
            SET face_id = person.face_id, face_status = 'known'
            FROM persons AS person
            WHERE event.person_id = person.id AND event.recognized = true
            """
        )
    )
    op.create_index(
        "ix_recognition_events_face_id",
        "recognition_events",
        ["face_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_recognition_events_face_id", table_name="recognition_events")
    op.drop_column("recognition_events", "face_status")
    op.drop_column("recognition_events", "face_id")
    op.drop_index(
        "ix_anonymous_face_embeddings_identity_id",
        table_name="anonymous_face_embeddings",
    )
    op.drop_table("anonymous_face_embeddings")
    op.drop_table("anonymous_identities")
    op.drop_constraint("uq_persons_face_id", "persons", type_="unique")
    op.drop_column("persons", "face_id")
