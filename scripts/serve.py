"""Start the replay page.

Usage:
    python scripts/serve.py            # then open http://localhost:8000
    python scripts/serve.py --port 8080

Needs generated matches in data/ (scripts/generate_match.py). Stories from
scripts/run_pipeline.py are shown straight away; the page can also run the agents live.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    a = p.parse_args()
    import uvicorn
    print(f"Replay page: http://localhost:{a.port}")
    uvicorn.run("whatelse.app:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
