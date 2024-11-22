import uvicorn
import os

if __name__ == "__main__":
    # Get absolute paths for api and ml directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.join(base_dir, "api")
    ml_dir = os.path.join(base_dir, "ml")

    uvicorn.run(
        "api.main:app",
        reload=True,
        reload_dirs=[api_dir, ml_dir],  # Use absolute paths
        reload_excludes=[
            "*__pycache__*",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*plotly*",
            "*site-packages*",
            "*thunderai-env*",
            "*scipy*",
            "*torch*",
            "*numpy*",
            "*pandas*"
        ],
        reload_includes=["*.py"],  # Only watch Python files
        log_level="debug",
        host="127.0.0.1",
        port=8000,
        workers=1  # Use single worker in development
    ) 