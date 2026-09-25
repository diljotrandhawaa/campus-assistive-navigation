#!/usr/bin/env python3
"""Start this copy's YOLO UI, without changing the live-vlm-webui command."""
from pathlib import Path
import sys

# Import only the add-on next to this file, even if another WebUI is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_live.server import main

if __name__ == "__main__":
    main()
