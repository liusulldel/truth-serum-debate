"""Run one small live BTS debate round.

Set ANTHROPIC_API_KEY in the environment or in a local .env file before running.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.debate import DebateConfig, run_debate


def _load_dotenv() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    _load_dotenv()
    question = "Water boils at 100 degrees Celsius at sea level."
    result = run_debate(question, cfg=DebateConfig())
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
