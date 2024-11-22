import os
import sys

def add_project_root_to_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

if __name__ == "__main__":
    add_project_root_to_path() 