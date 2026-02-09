#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


def build_filename_index(src_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in src_dir.glob("*.html"):
        stem = p.stem
        idx.setdefault(stem, p)
        idx.setdefault(stem.replace("_", " "), p)
        # idx.setdefault(stem.replace(" ", "_"), p)
    return idx


def main() -> int:
    filter_file = Path("kb-filter.txt")

    src_dir = Path("wiki-src")
    dst_dir = Path("knowledge_base_html_src")

    dst_dir.mkdir(parents=True, exist_ok=True)

    filter = filter_file.read_text(encoding="utf-8").splitlines()

    file_index = build_filename_index(src_dir)

    not_found: list[str] = []
    found: list[str] = []
    processed = 0
    for title in filter:
        p = file_index.get(title)
        if p is None:
            not_found.append(title)
            continue
        else:
            found.append(title)

        shutil.copy2(p, dst_dir)
        processed += 1

    if not_found:
        print("\nNot found titles:")
        for t in not_found:
            print(f"- {t}")
        print(f"\nFound titles {len(found)}:")
        for t in found:
            print(f"- {t}")
    else:
        print("\nAll titles found.")
    print(f"Processed files: {processed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
