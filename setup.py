from setuptools import setup, find_packages

setup(
    name="thunderai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.27.0",
        "python-jose[cryptography]",
        "passlib[bcrypt]>=1.7.4",
        "bcrypt>=4.0.1",
        "python-multipart",
        "sqlalchemy>=2.0.25",
        "psycopg2-binary>=2.9.9",
        "redis>=5.0.1",
        "prometheus-client>=0.19.0",
        "plotly",
        "networkx",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.2",
        "typing-extensions>=4.6.0",
        "pydantic>=2.0.0",
        "pydantic[email]",
        "email-validator>=2.0.0",
        "mlflow>=2.9.2",
        "fastapi-cache2>=0.2.1",
        "pydantic-settings>=2.0.0",
        "httpx>=0.24.0"
    ],
    python_requires=">=3.10,<3.12",
    extras_require={
        'ml': [
            'torch',
            'torchvision',
            'torchaudio',
            'transformers>=4.36.2',
        ],
        'apple': [
            'tensorflow-macos',
            'tensorflow-metal',
        ],
        'nvidia': [
            'tensorflow',
            'cuda-python',
        ]
    }
)