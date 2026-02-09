from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md

ALLOWED_TAGS = {"div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "li", "br"}

# --- Markdown cleanup helpers ---
_IMG_INLINE_RE = re.compile(r"!\[[^\]]*]\([^)]*\)")
_LINK_INLINE_RE = re.compile(r"\[([^\]]+)]\([^)]*\)")
_LINK_REF_DEF_RE = re.compile(r"(?m)^\s*\[[^\]]+]:\s+\S+.*$")


def extract_container(soup: BeautifulSoup) -> Tag | None:
    # div with id="content" and class contains "page-content"
    return soup.find("div", id="content")


def clean_content(container: Tag) -> str:
    # 1) Remove tables entirely (including their text)
    for t in container.find_all("table"):
        t.decompose()

    # 2) Remove scripts/styles etc.
    for t in container.find_all(["script", "style", "noscript"]):
        t.decompose()

    # 3) Keep text when removing links
    for a in container.find_all("a"):
        a.unwrap()

    # 4) Remove/unwrap all tags except allowed
    # Iterate over a snapshot list because we mutate the tree
    for tag in list(container.find_all(True)):
        if tag.name in ALLOWED_TAGS:
            # Optional: clean attributes to avoid junk
            # Keep only id/class if you want; currently removes all attrs.
            tag.attrs = {}
        else:
            # unwrap keeps children/text, removes the tag itself
            tag.unwrap()

    # Return the extracted block (inner HTML only)
    text = container.decode_contents()
    return text


def strip_links_and_images(md_text: str) -> str:
    # Remove inline images completely
    md_text = _IMG_INLINE_RE.sub("", md_text)

    # Replace inline links with just the link text
    # [text](url) -> text
    md_text = _LINK_INLINE_RE.sub(r"\1", md_text)

    # Remove reference-style link definitions
    md_text = _LINK_REF_DEF_RE.sub("", md_text)

    # Tidy multiple blank lines caused by removals
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)

    md_text = md_text.replace("[]", "")

    return md_text


def make_replacer(mapping: dict[str, str]):
    # Replace longer keys first to avoid partial overlaps
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in keys))

    def repl(m: re.Match) -> str:
        return mapping[m.group(0)]

    return pattern, repl


def prepare_md():
    src_dir = Path("knowledge_base_html_src")
    dst_dir = Path("knowledge_base_md_src")
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        print(f"ERROR: source directory not found: {src_dir}", file=sys.stderr)
        return 2

    files = sorted(src_dir.glob("*.html"))
    if not files:
        print(f"No .html files found in {src_dir}")
        return 0

    written = 0
    missing = 0

    for src_path in files:
        raw = src_path.read_text(encoding="utf-8", errors="replace")
        raw = raw.replace('†', ' ')
        soup = BeautifulSoup(raw, "html.parser")

        container = extract_container(soup)
        if container is None:
            print(f"[WARN] container not found in: {src_path.name}")
            missing += 1
            continue

        cleaned_html = clean_content(container)
        out_path = dst_dir / f"{src_path.stem}.html"
        out_path.write_text(cleaned_html, encoding="utf-8", newline="\n")

        md_text = md(
            cleaned_html,
            heading_style="ATX",
            bullets="-",
        )

        # Strip links and images from Markdown
        md_text = strip_links_and_images(md_text)
        md_text = md_text.replace("\xa0", " ").replace("\u00A0", " ")

        # Normalize line endings and ensure trailing newline
        md_text = md_text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"

        out_path = dst_dir / f"{src_path.stem}.md"
        out_path.write_text(md_text, encoding="utf-8", newline="\n")
        written += 1

    print(f"Prepare done. Processed files: {written}")
    return None


def terms_replace():
    mapping_file = Path("terms_map.json")
    mapping = {}
    with open(mapping_file) as f_in:
        mapping = json.load(f_in)

    src_dir = Path("knowledge_base_md_src")
    dst_dir = Path("knowledge_base")

    dst_dir.mkdir(parents=True, exist_ok=True)

    pattern, repl = make_replacer(mapping)
    processed = 0
    for p in src_dir.glob("*.md"):
        text = p.read_text(encoding="utf-8", errors="strict")
        new_text = pattern.sub(repl, text)
        file_name = pattern.sub(repl, p.stem).replace(" ", "_")

        out_path = dst_dir / f"{file_name}.md"
        out_path.write_text(new_text, encoding="utf-8", newline="\n")
        processed += 1

    print(f"Terms replace done. Processed files: {processed}")
    return None


if __name__ == "__main__":
    prepare_md()
    terms_replace()
