"""Add authentication, roles, sessions and data ownership.

Revision ID: 012
Revises: 011
Create Date: 2026-08-27
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=150), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=20), server_default="user", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("role IN ('admin', 'user')", name="ck_users_role"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("uq_users_username_lower", "users", [sa.text("lower(username)")], unique=True)
    op.create_index("uq_users_email_lower", "users", [sa.text("lower(email)")], unique=True)

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("refresh_token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("refresh_token_hash"),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])
    op.create_index("ix_auth_sessions_expires_at", "auth_sessions", ["expires_at"])

    op.add_column("persons", sa.Column("owner_user_id", sa.Uuid(), nullable=True))
    op.add_column("persons", sa.Column("is_global", sa.Boolean(), server_default=sa.false(), nullable=False))
    op.create_foreign_key("fk_persons_owner_user_id", "persons", "users", ["owner_user_id"], ["id"], ondelete="SET NULL")
    op.create_index("ix_persons_owner_user_id", "persons", ["owner_user_id"])
    op.execute("UPDATE persons SET is_global = true")

    op.add_column("anonymous_identities", sa.Column("owner_user_id", sa.Uuid(), nullable=True))
    op.create_foreign_key("fk_anonymous_identities_owner_user_id", "anonymous_identities", "users", ["owner_user_id"], ["id"], ondelete="CASCADE")
    op.create_index("ix_anonymous_identities_owner_user_id", "anonymous_identities", ["owner_user_id"])

    op.add_column("recognition_processes", sa.Column("owner_user_id", sa.Uuid(), nullable=True))
    op.create_foreign_key("fk_recognition_processes_owner_user_id", "recognition_processes", "users", ["owner_user_id"], ["id"], ondelete="SET NULL")
    op.create_index("ix_recognition_processes_owner_user_id", "recognition_processes", ["owner_user_id"])


def downgrade() -> None:
    op.drop_index("ix_recognition_processes_owner_user_id", table_name="recognition_processes")
    op.drop_constraint("fk_recognition_processes_owner_user_id", "recognition_processes", type_="foreignkey")
    op.drop_column("recognition_processes", "owner_user_id")
    op.drop_index("ix_anonymous_identities_owner_user_id", table_name="anonymous_identities")
    op.drop_constraint("fk_anonymous_identities_owner_user_id", "anonymous_identities", type_="foreignkey")
    op.drop_column("anonymous_identities", "owner_user_id")
    op.drop_index("ix_persons_owner_user_id", table_name="persons")
    op.drop_constraint("fk_persons_owner_user_id", "persons", type_="foreignkey")
    op.drop_column("persons", "is_global")
    op.drop_column("persons", "owner_user_id")
    op.drop_index("ix_auth_sessions_expires_at", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_index("uq_users_email_lower", table_name="users")
    op.drop_index("uq_users_username_lower", table_name="users")
    op.drop_table("users")
