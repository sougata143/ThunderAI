"""update user table

Revision ID: xxxx
Revises: previous_revision
Create Date: 2024-03-22

"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    # Drop columns that don't exist in new schema
    op.drop_column('users', 'first_name')
    op.drop_column('users', 'last_name')
    op.drop_column('users', 'organization')
    op.drop_column('users', 'job_title')
    op.drop_column('users', 'settings')
    
    # Add new column
    op.add_column('users', sa.Column('full_name', sa.String(), nullable=True))

def downgrade():
    # Add back original columns
    op.add_column('users', sa.Column('first_name', sa.String(), nullable=True))
    op.add_column('users', sa.Column('last_name', sa.String(), nullable=True))
    op.add_column('users', sa.Column('organization', sa.String(), nullable=True))
    op.add_column('users', sa.Column('job_title', sa.String(), nullable=True))
    op.add_column('users', sa.Column('settings', sa.JSON(), nullable=True))
    
    # Remove new column
    op.drop_column('users', 'full_name') 