#!/usr/bin/env python3
"""
Apply title renames to a subset of markdown files.

Inputs:
- a mapping file (lines like: "Agent Smith -> Ivan Michailo")
- a source directory with .md files
- a destination directory

Behavior:
1) Only processes files whose stem matches a title in the mapping, with filename variants:
   - Exact: "Trinity.md"
   - Underscore: "Agent_Smith.md" (spaces <-> underscores)
2) Replaces occurrences in file content using the mapping (all rules).
3) Writes the processed file to destination dir with the SAME filename as the source file.
4) Prints not-found titles (no matching .md file found in source dir).
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


def parse_mapping(mapping_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for i, raw in enumerate(mapping_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "->" not in line:
            print(f"[WARN] Skipping line {i} (no '->'): {raw!r}", file=sys.stderr)
            continue
        src, dst = (part.strip() for part in line.split("->", 1))
        if not src:
            print(f"[WARN] Skipping line {i} (empty source): {raw!r}", file=sys.stderr)
            continue
        mapping[src] = dst
    return mapping


def build_filename_index(src_dir: Path) -> dict[str, Path]:
    """
    Index markdown files by stem and also by "stem with spaces" <-> underscores variant.
    This lets us match:
      title "Agent Smith" -> file "Agent_Smith.md"
      title "Trinity" -> file "Trinity.md"
    """
    idx: dict[str, Path] = {}
    for p in src_dir.glob("*.md"):
        stem = p.stem
        idx.setdefault(stem, p)
        idx.setdefault(stem.replace("_", " "), p)  # Agent_Smith -> Agent Smith
        idx.setdefault(stem.replace(" ", "_"), p)  # Agent Smith -> Agent_Smith
    return idx


def make_replacer(mapping: dict[str, str]):
    # Replace longer keys first to avoid partial overlaps
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in keys))

    def repl(m: re.Match) -> str:
        return mapping[m.group(0)]

    return pattern, repl


def main() -> int:
    # You can change defaults here if you want.
    mapping_file = Path("titles.txt")  # file with "Old -> New"
    src_dir = Path("md-dest")
    dst_dir = Path("knowledge_base")

    # Or allow CLI args: script.py titles.txt md-dest knowledge_base
    if len(sys.argv) == 4:
        mapping_file = Path(sys.argv[1])
        src_dir = Path(sys.argv[2])
        dst_dir = Path(sys.argv[3])
    elif len(sys.argv) != 1:
        print("Usage: python rename_apply.py <titles.txt> <md-dest> <knowledge_base>", file=sys.stderr)
        return 2

    if not mapping_file.exists():
        print(f"[ERROR] Mapping file not found: {mapping_file}", file=sys.stderr)
        return 1
    if not src_dir.exists():
        print(f"[ERROR] Source dir not found: {src_dir}", file=sys.stderr)
        return 1

    dst_dir.mkdir(parents=True, exist_ok=True)

    mapping = parse_mapping(mapping_file)
    if not mapping:
        print("[ERROR] No valid mappings found.", file=sys.stderr)
        return 1

    file_index = build_filename_index(src_dir)
    pattern, repl = make_replacer(mapping)

    not_found: list[str] = []
    processed = 0

    for title in mapping.keys():
        p = file_index.get(title)
        if p is None:
            not_found.append(title)
            continue

        text = p.read_text(encoding="utf-8", errors="strict")
        new_text = pattern.sub(repl, text)

        out_path = dst_dir / p.name
        out_path.write_text(new_text, encoding="utf-8", newline="\n")
        processed += 1

    print(f"Processed files: {processed}")
    if not_found:
        print("\nNot found titles:")
        for t in not_found:
            print(f"- {t}")
    else:
        print("\nAll titles found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
