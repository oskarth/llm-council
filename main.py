"""Entry point for llm-council: starts both backend and frontend."""

import os
import signal
import subprocess
import sys

import uvicorn


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_dir, "frontend")

    # Start frontend dev server
    frontend_proc = None
    if os.path.isdir(frontend_dir):
        print("Starting frontend dev server...")
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def cleanup(signum, frame):
        if frontend_proc:
            frontend_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        print("Starting backend on http://localhost:8001...")
        uvicorn.run("backend.main:app", host="0.0.0.0", port=8001)
    finally:
        if frontend_proc:
            frontend_proc.terminate()


if __name__ == "__main__":
    main()
