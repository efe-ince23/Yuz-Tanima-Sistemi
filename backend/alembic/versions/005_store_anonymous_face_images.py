"""Store cropped images for anonymous face samples.

Revision ID: 005
Revises: 004
Create Date: 2026-08-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "anonymous_face_embeddings",
        sa.Column("image_path", sa.String(length=500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("anonymous_face_embeddings", "image_path")
