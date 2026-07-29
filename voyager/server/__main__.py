"""Run the Voyager replay application."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "voyager.server.app:create_app",
        factory=True,
        host=os.environ.get("VOYAGER_HOST", "127.0.0.1"),
        port=int(os.environ.get("VOYAGER_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
