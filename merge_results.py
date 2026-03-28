"""Merge results.tsv files from multiple agent worktrees into the main repo.

Usage:
    uv run merge_results.py ../cuda-evolve-matmul ../cuda-evolve-rms-norm

This reads results.tsv from each worktree directory, deduplicates rows by
experiment_id, and writes the merged result to results.tsv in the current
working directory.
"""

import sys
from pathlib import Path


HEADER = "experiment_id\thypothesis\tcorrectness\ttime_ms\tthroughput\tpeak_vram_mb\tkept\n"


def load_rows(path: Path) -> list[str]:
    if not path.exists():
        print(f"  [skip] {path} does not exist")
        return []
    lines = path.read_text().strip().split("\n")
    if len(lines) <= 1:
        print(f"  [skip] {path} has no data rows")
        return []
    rows = lines[1:]
    print(f"  [load] {path}: {len(rows)} rows")
    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run merge_results.py <worktree_dir> [<worktree_dir> ...]")
        print("Merges results.tsv from each worktree into ./results.tsv")
        sys.exit(1)

    worktree_dirs = [Path(d) for d in sys.argv[1:]]
    local_results = Path("results.tsv")

    all_rows: list[str] = []

    if local_results.exists():
        print("Loading local results.tsv:")
        all_rows.extend(load_rows(local_results))

    for wd in worktree_dirs:
        results_path = wd / "results.tsv"
        print(f"Loading from {wd}:")
        all_rows.extend(load_rows(results_path))

    seen_ids: set[str] = set()
    unique_rows: list[str] = []
    for row in all_rows:
        exp_id = row.split("\t")[0] if "\t" in row else row
        if exp_id not in seen_ids:
            seen_ids.add(exp_id)
            unique_rows.append(row)

    local_results.write_text(HEADER + "\n".join(unique_rows) + "\n")
    print(f"\nMerged {len(unique_rows)} unique rows into {local_results}")


if __name__ == "__main__":
    main()
