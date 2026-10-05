"""add title to documents

Revision ID: fe80d49f8d69
Revises: 70c8891613ca
Create Date: 2026-10-05 23:53:40.148538

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fe80d49f8d69'
down_revision: Union[str, Sequence[str], None] = '70c8891613ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('documents', sa.Column('title', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('documents', 'title')
