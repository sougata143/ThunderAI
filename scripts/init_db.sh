#!/bin/bash

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create database if it doesn't exist
PGPASSWORD=mypass psql -h localhost -U thunderai_user -d postgres -c "CREATE DATABASE thunderai;" || true

# Initialize alembic if not already initialized
if [ ! -d "alembic/versions" ]; then
    echo "Initializing alembic..."
    alembic init alembic
fi

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Initialize database with test data
echo "Initializing database with test data..."
python -c "
from db.session import SessionLocal
from db.init_db import init_db

db = SessionLocal()
init_db(db)
"

echo "Database initialization completed!" 