import os
from pathlib import Path

PATHS = {
    "data": Path("data"),
    "out": Path("out")
}

def create_paths(paths):
    for _, path in paths.items():
        if not path.is_dir():
            os.system(f"mkdir {path}")
