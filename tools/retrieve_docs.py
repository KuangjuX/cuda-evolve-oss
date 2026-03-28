#!/usr/bin/env python3
"""
retrieve_docs.py -- Knowledge retrieval for cuda-evolve agent loop.

Searches the docs/ directory and CUDA_OPTIMIZATION.md for sections relevant
to a given query. Uses TF-IDF-like keyword matching over markdown sections.

Usage:
  uv run tools/retrieve_docs.py "long scoreboard stall"
  uv run tools/retrieve_docs.py "register pressure reduction"
  uv run tools/retrieve_docs.py "L2 cache hit rate low"
  uv run tools/retrieve_docs.py --list                          # list all indexed sections
  uv run tools/retrieve_docs.py --top 3 "memory coalescing"     # top 3 results
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
OPTIMIZATION_FILE = ROOT / "CUDA_OPTIMIZATION.md"


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return [w for w in text.split() if len(w) > 1]


def _parse_sections(filepath: Path) -> list[dict]:
    """Split a markdown file into sections by ## headings."""
    if not filepath.exists():
        return []

    content = filepath.read_text(encoding="utf-8")
    sections = []
    current_title = filepath.stem
    current_lines: list[str] = []

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({
                        "file": str(filepath.relative_to(ROOT)),
                        "title": current_title,
                        "body": body,
                        "tokens": _tokenize(body + " " + current_title),
                    })
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({
                "file": str(filepath.relative_to(ROOT)),
                "title": current_title,
                "body": body,
                "tokens": _tokenize(body + " " + current_title),
            })

    return sections


def build_index() -> list[dict]:
    """Build index over all docs/ files and CUDA_OPTIMIZATION.md."""
    sections = []

    if DOCS_DIR.exists():
        for md_file in sorted(DOCS_DIR.glob("*.md")):
            sections.extend(_parse_sections(md_file))

    if OPTIMIZATION_FILE.exists():
        sections.extend(_parse_sections(OPTIMIZATION_FILE))

    return sections


def search(query: str, sections: list[dict], top_k: int = 5) -> list[tuple[float, dict]]:
    """TF-IDF-style search over indexed sections."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    n_docs = len(sections)
    if n_docs == 0:
        return []

    doc_freq: Counter[str] = Counter()
    for sec in sections:
        unique_tokens = set(sec["tokens"])
        for t in unique_tokens:
            doc_freq[t] += 1

    results: list[tuple[float, dict]] = []
    for sec in sections:
        token_freq = Counter(sec["tokens"])
        doc_len = len(sec["tokens"]) or 1
        score = 0.0

        for qt in query_tokens:
            tf = token_freq.get(qt, 0) / doc_len
            df = doc_freq.get(qt, 0)
            idf = math.log((n_docs + 1) / (df + 1)) + 1
            score += tf * idf

            for st in token_freq:
                if qt in st or st in qt:
                    partial_tf = token_freq[st] / doc_len * 0.3
                    score += partial_tf * idf

        if score > 0:
            results.append((score, sec))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge retrieval for cuda-evolve: search docs by keyword"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Search query (e.g., 'long scoreboard stall mitigation')",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all indexed sections",
    )
    args = parser.parse_args()

    sections = build_index()

    if args.list:
        print(f"=== INDEXED SECTIONS ({len(sections)} total) ===")
        for sec in sections:
            n_tokens = len(sec["tokens"])
            print(f"  [{sec['file']}] {sec['title']} ({n_tokens} tokens)")
        print("=== END ===")
        return

    query = " ".join(args.query)
    if not query:
        parser.print_help()
        return

    results = search(query, sections, top_k=args.top)

    print(f"\n=== DOCS RETRIEVAL: \"{query}\" ({len(results)} results) ===\n")

    if not results:
        print("No matching sections found.")
        print("Available docs:")
        for sec in sections[:10]:
            print(f"  [{sec['file']}] {sec['title']}")
        return

    for i, (score, sec) in enumerate(results):
        print(f"--- Result {i+1} (score: {score:.3f}) [{sec['file']}] ---")
        print(f"## {sec['title']}\n")
        body = sec["body"]
        if len(body) > 2000:
            body = body[:2000] + "\n... (truncated, see full file)"
        print(body)
        print()

    print("=== END DOCS RETRIEVAL ===")


if __name__ == "__main__":
    main()
