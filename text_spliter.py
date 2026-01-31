import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Section:
    title: str
    section_path: List[str]  # e.g. ["History"] or ["Leadership", "Head of State"]
    text: str  # section content (no headings)
    start_line: int
    end_line: int


HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<name>.+?)\s*$")


def parse_markdown_sections(md: str, default_title: str) -> List[Section]:
    """
    Parse markdown text into hierarchical sections using #..###### headings.
    Returns list of Section objects, each containing section_path + body text.
    """
    lines = md.splitlines()
    title = default_title

    # Stack holds tuples: (level, heading_name)
    stack: List[Tuple[int, str]] = []

    # Accumulator for current section body
    current_body: List[str] = []
    current_start_line = 1

    sections: List[Section] = []

    def flush(end_line: int):
        nonlocal current_body, current_start_line
        body = "\n".join(current_body).strip()
        # Only emit if there is meaningful content
        if body:
            sections.append(
                Section(
                    title=title,
                    section_path=[h for _, h in stack],  # path in order
                    text=body,
                    start_line=current_start_line,
                    end_line=end_line,
                )
            )
        current_body = []

    for i, line in enumerate(lines, start=1):
        m = HEADING_RE.match(line)
        if m:
            # finish previous section up to line i-1
            flush(end_line=i - 1)

            level = len(m.group("hashes"))
            name = m.group("name").strip()

            # If first level-1 heading appears, treat as document title
            if level == 1 and not stack:
                title = name
                # reset to "top-level" (no section path yet)
                stack = []
            else:
                # Pop headings until parent level < current level
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, name))

            current_start_line = i + 1  # content starts after heading
        else:
            current_body.append(line)

    # flush tail
    flush(end_line=len(lines))
    return sections


# ---------- OPTIONAL: build embedding-ready chunks ----------
def inject_context(title: str, section_path: List[str], chunk: str) -> str:
    sec = " > ".join(section_path) if section_path else "(no section)"
    return f"Title: {title}\nSection: {sec}\n\n{chunk}".strip()


def split_section_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """
    Lightweight, dependency-free chunker by characters with overlap.
    You can replace this with LangChain's RecursiveCharacterTextSplitter if you prefer.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks


def make_embedding_docs(md: str, default_title: str = "Untitled") -> List[dict]:
    """
    Returns list of dicts:
      {
        "text": embedding_input_text,
        "meta": { title, section_path, start_line, end_line, chunk_index }
      }
    """
    sections = parse_markdown_sections(md, default_title=default_title)

    docs = []
    for s in sections:
        section_chunks = split_section_text(s.text, chunk_size=1200, overlap=150)
        for idx, ch in enumerate(section_chunks):
            docs.append({
                "text": inject_context(s.title, s.section_path, ch),
                "meta": {
                    "title": s.title,
                    "section_path": s.section_path,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "chunk_index": idx,
                }
            })
    return docs
