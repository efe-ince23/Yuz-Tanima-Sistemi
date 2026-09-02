"""Add resumable dataset import tracking.

Revision ID: 010
Revises: 009
Create Date: 2026-08-21
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("persons", sa.Column("source", sa.String(length=50), nullable=True))
    op.add_column(
        "persons", sa.Column("external_id", sa.String(length=255), nullable=True)
    )
    op.create_index(
        "uq_persons_source_external_id",
        "persons",
        ["source", "external_id"],
        unique=True,
    )
    op.create_table(
        "dataset_import_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_name", sa.String(length=50), nullable=False),
        sa.Column("external_identity", sa.String(length=255), nullable=False),
        sa.Column("source_path", sa.String(length=500), nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=True),
        sa.Column("face_image_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("image_path", sa.String(length=500), nullable=True),
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
        sa.ForeignKeyConstraint(["face_image_id"], ["face_images.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["person_id"], ["persons.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_dataset_import_items_dataset_status",
        "dataset_import_items",
        ["dataset_name", "status"],
    )
    op.create_index(
        "uq_dataset_import_items_dataset_source_path",
        "dataset_import_items",
        ["dataset_name", "source_path"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_dataset_import_items_dataset_source_path",
        table_name="dataset_import_items",
    )
    op.drop_index(
        "ix_dataset_import_items_dataset_status", table_name="dataset_import_items"
    )
    op.drop_table("dataset_import_items")
    op.drop_index("uq_persons_source_external_id", table_name="persons")
    op.drop_column("persons", "external_id")
    op.drop_column("persons", "source")
