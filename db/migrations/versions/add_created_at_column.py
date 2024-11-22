"""add created_at column

Revision ID: add_created_at_column
Revises: xxxx
Create Date: 2024-03-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func

def upgrade():
    # Add created_at column with server_default
    op.add_column('users', 
        sa.Column('created_at', 
                  sa.DateTime(timezone=True), 
                  server_default=func.now(),
                  nullable=False)
    )

def downgrade():
    # Remove created_at column
    op.drop_column('users', 'created_at') 