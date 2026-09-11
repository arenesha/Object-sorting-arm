"""SORT | ARM AI - Dashboard Launcher

Starts the web-based vision, model manager, and arm control dashboard.

Usage:
    python dashboard.py               # Runs on port 5050 (default)
    python dashboard.py --port 5000   # Run on port 5000
    python dashboard.py --host 0.0.0.0 # Accessible over local network / Jetson IP
"""

import os
import sys
import argparse

# Add dashboard directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard")))
from app import app


def main():
    parser = argparse.ArgumentParser(description="SORT | ARM AI Web Dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Server port (default: 5050)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    # Pre-initialize camera device
    from app import STATE
    STATE.init_camera(args.camera)

    print("=" * 65)
    print("                 SORT | ARM AI DASHBOARD                  ")
    print(f"       Local URL:   http://localhost:{args.port}         ")
    print(f"       Network URL: http://{args.host}:{args.port}       ")
    print(f"       Camera:      Device Index {args.camera}           ")
    print("=" * 65)

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
