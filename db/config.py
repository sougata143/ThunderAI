import os
from sqlalchemy.engine.url import URL

# Database configuration
DB_PARAMS = {
    'drivername': 'postgresql',
    'username': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'thunderai')
}

DATABASE_URL = URL.create(**DB_PARAMS) 