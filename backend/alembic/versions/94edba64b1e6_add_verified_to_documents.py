"""add verified to documents

Revision ID: 94edba64b1e6
Revises: fe80d49f8d69
Create Date: 2026-10-06 17:57:48.043049

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '94edba64b1e6'
down_revision: Union[str, Sequence[str], None] = 'fe80d49f8d69'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'documents',
        sa.Column('verified', sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('documents', 'verified')
