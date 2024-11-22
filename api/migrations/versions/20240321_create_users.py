"""create users table

Revision ID: 20240321_create_users
Revises: 
Create Date: 2024-03-21
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func

# revision identifiers
revision = '20240321_create_users'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(), nullable=True),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('hashed_password', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=func.now()),
        sa.Column('full_name', sa.String(), nullable=True),
        sa.Column('organization', sa.String(), nullable=True),
        sa.Column('job_title', sa.String(), nullable=True),
        sa.Column('settings', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    # Create indexes
    op.create_index('ix_users_id', 'users', ['id'], unique=False)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

def downgrade():
    # Drop indexes
    op.drop_index('ix_users_email')
    op.drop_index('ix_users_username')
    op.drop_index('ix_users_id')
    # Drop table
    op.drop_table('users') 