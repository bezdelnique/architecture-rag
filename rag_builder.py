from __future__ import annotations

from pathlib import Path
from text_spliter import make_embedding_docs, parse_markdown_sections
import re
import sys
import json

from vector_db import upsert_docs_to_chroma


def main() -> int:
    dst_dir = Path("knowledge_base")

    if not dst_dir.exists():
        print(f"[ERROR] Source dir not found: {dst_dir}", file=sys.stderr)
        return 1

    # idx: dict[str, Path] = {}
    for p in dst_dir.glob("*.md"):

        if not any(name in p.name for name in {"Trinity", "Neo"}):
            continue

        print(f"[INFO] Processing {p.name}")
        title = p.stem.replace("_", " ")
        text = p.read_text(encoding="utf-8", errors="strict")
        collection = upsert_docs_to_chroma(make_embedding_docs(text, title))
        for row in collection:
            print(row)


    return 0

if __name__ == "__main__":
    raise SystemExit(main())
