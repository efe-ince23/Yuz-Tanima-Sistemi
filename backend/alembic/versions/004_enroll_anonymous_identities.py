"""Allow anonymous identities to be enrolled as known persons.

Revision ID: 004
Revises: 003
Create Date: 2026-08-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "anonymous_identities",
        sa.Column("person_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_anonymous_identities_person_id",
        "anonymous_identities",
        "persons",
        ["person_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_unique_constraint(
        "uq_anonymous_identities_person_id",
        "anonymous_identities",
        ["person_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_anonymous_identities_person_id",
        "anonymous_identities",
        type_="unique",
    )
    op.drop_constraint(
        "fk_anonymous_identities_person_id",
        "anonymous_identities",
        type_="foreignkey",
    )
    op.drop_column("anonymous_identities", "person_id")
