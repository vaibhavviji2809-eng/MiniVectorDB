"""Console dashboard entrypoint."""

from __future__ import annotations

from ..core import MiniVectorDB
from .report import to_json


def main() -> None:
    db = MiniVectorDB()
    print(to_json(db))


if __name__ == "__main__":
    main()

