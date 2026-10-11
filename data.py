"""Parse the LARA TripAdvisor hotel_<ID>_parsed.txt files into a clean ratings table.

Usage:
    python data.py [--data-root .] [--out data/ratings.parquet]

Writes data/ratings.parquet, data/user_map.csv, data/item_map.csv and
results/data_stats.md (Tables 1 and 2 of the paper, with diffs).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paper_values import PAPER_TABLE1, PAPER_TABLE2

log = logging.getLogger("data")

# Order of the 8 values on the <Rating> line.
RATING_COLS = ["overall", "value", "rooms", "location", "cleanliness", "checkin", "service", "business"]
CRITERIA = RATING_COLS[1:]
CRITERIA_LABELS = {
    "overall": "Overall",
    "value": "Value",
    "rooms": "Rooms",
    "location": "Location",
    "cleanliness": "Cleanliness",
    "checkin": "Check in / front desk",
    "service": "Service",
    "business": "Business service",
}
ANONYMOUS_AUTHOR = "A TripAdvisor Member"

FILE_RE = re.compile(r"^hotel_(\d+)_parsed\.txt$")
# A review starts at each <Author> tag; the other tags are searched inside that chunk.
AUTHOR_SPLIT_RE = re.compile(r"<Author>")
CONTENT_RE = re.compile(r"<Content>(.*?)(?=<Date>|<Rating>|\Z)", re.DOTALL)
DATE_RE = re.compile(r"<Date>([^\n]*)")
RATING_RE = re.compile(r"<Rating>([^\n]*)")

SKIP_DIRS = {".venv", "venv", ".git", "results", "data", "__pycache__", "node_modules"}


def parse_rating_line(line: str | None) -> list[int] | None:
    """Return the list of ints on a <Rating> line, or None if any token is not an int."""
    if line is None:
        return None
    tokens = line.strip().split("\t")  # strip() drops the trailing tab and \r
    try:
        return [int(t) for t in tokens]
    except ValueError:
        return None


def parse_file(path: str | Path) -> list[dict]:
    """Parse one hotel file into a list of raw review dicts (no filtering)."""
    path = Path(path)
    m = FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"not a hotel_<ID>_parsed.txt file: {path}")
    hotel_id = m.group(1)
    text = path.read_bytes().decode("utf-8", errors="replace")
    reviews = []
    for chunk in AUTHOR_SPLIT_RE.split(text)[1:]:
        # Author is the rest of the tag's line; a stray \r ends it too.
        author = re.split(r"[\r\n]", chunk, maxsplit=1)[0].strip()
        content = CONTENT_RE.search(chunk)
        date = DATE_RE.search(chunk)
        rating = RATING_RE.search(chunk)
        reviews.append(
            {
                "user": author,
                "hotel": hotel_id,
                "content": content.group(1).replace("\r", " ").strip() if content else "",
                "date": date.group(1).strip() if date else "",
                "ratings": parse_rating_line(rating.group(1) if rating else None),
            }
        )
    return reviews


def find_files(root: str | Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        out.extend(Path(dirpath) / f for f in filenames if FILE_RE.match(f))
    return sorted(out, key=lambda p: (int(FILE_RE.match(p.name).group(1)), str(p)))


def filter_reviews(raw: list[dict]) -> pd.DataFrame:
    """Apply the 4 filtering steps (in order) and log the row count after each."""
    log.info("parsed reviews: %d", len(raw))

    rows = [r for r in raw if r["ratings"] is not None and len(r["ratings"]) == 8]
    log.info("after 1. exactly 8 rating values:        %d (-%d)", len(rows), len(raw) - len(rows))

    n = len(rows)
    rows = [r for r in rows if all(1 <= v <= 5 for v in r["ratings"])]
    log.info("after 2. no -1/0 (all ratings in 1..5):  %d (-%d)", len(rows), n - len(rows))

    n = len(rows)
    rows = [r for r in rows if r["user"] != ANONYMOUS_AUTHOR]
    log.info("after 3. drop '%s':        %d (-%d)", ANONYMOUS_AUTHOR, len(rows), n - len(rows))

    df = pd.DataFrame(
        {
            "user": [r["user"] for r in rows],
            "hotel": [r["hotel"] for r in rows],
            **{c: np.array([r["ratings"][k] for r in rows], dtype=np.int8) for k, c in enumerate(RATING_COLS)},
        }
    )
    n = len(df)
    df = df.drop_duplicates(subset=["user", "hotel"], keep="first").reset_index(drop=True)
    log.info("after 4. dedupe (user, hotel):           %d (-%d)", len(df), n - len(df))
    return df


def index_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map users/hotels to contiguous int indices (sorted, so deterministic)."""
    user_idx, users = pd.factorize(df["user"], sort=True)
    item_idx, items = pd.factorize(df["hotel"].astype(np.int64), sort=True)
    out = pd.DataFrame({"user_idx": user_idx.astype(np.int32), "item_idx": item_idx.astype(np.int32)})
    for c in RATING_COLS:
        out[c] = df[c].astype(np.int8).values
    user_map = pd.DataFrame({"user_idx": np.arange(len(users)), "user": users})
    item_map = pd.DataFrame({"item_idx": np.arange(len(items)), "hotel_id": items})
    return out, user_map, item_map


def stats_markdown(df: pd.DataFrame) -> str:
    n_users = int(df["user_idx"].nunique())
    n_items = int(df["item_idx"].nunique())
    n_ratings = len(df)
    sparsity = 1.0 - n_ratings / (n_users * n_items)
    ours1 = {"users": n_users, "hotels": n_items, "ratings": n_ratings, "sparsity": sparsity}

    lines = ["## Table 1 - dataset statistics", "", "| Field | Paper | Ours | Diff |", "|---|---:|---:|---:|"]
    for key, label in [("users", "Number of users"), ("hotels", "Number of hotels"), ("ratings", "Number of ratings")]:
        p, o = PAPER_TABLE1[key], ours1[key]
        lines.append(f"| {label} | {p:,} | {o:,} | {o - p:+,} |")
    p, o = PAPER_TABLE1["sparsity"], ours1["sparsity"]
    lines.append(f"| Data sparsity | {p:.2%} | {o:.4%} | {100 * (o - p):+.4f} pp |")

    lines += ["", "## Table 2 - ratings distribution (ours, with paper in brackets)", "",
              "| Rating | 1 | 2 | 3 | 4 | 5 |", "|---|---:|---:|---:|---:|---:|"]
    for c in CRITERIA + ["overall"]:
        counts = df[c].value_counts().reindex(range(1, 6), fill_value=0)
        cells = [f"{int(counts[v]):,} ({PAPER_TABLE2[c][v - 1]:,})" for v in range(1, 6)]
        lines.append(f"| {CRITERIA_LABELS[c]} | " + " | ".join(cells) + " |")

    lines += ["", "## Table 2 - diff (ours - paper)", "", "| Rating | 1 | 2 | 3 | 4 | 5 |", "|---|---:|---:|---:|---:|---:|"]
    for c in CRITERIA + ["overall"]:
        counts = df[c].value_counts().reindex(range(1, 6), fill_value=0)
        cells = [f"{int(counts[v]) - PAPER_TABLE2[c][v - 1]:+,}" for v in range(1, 6)]
        lines.append(f"| {CRITERIA_LABELS[c]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=".", help="directory scanned recursively for hotel_*_parsed.txt")
    ap.add_argument("--out", default="data/ratings.parquet")
    ap.add_argument("--stats-out", default="results/data_stats.md")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    files = find_files(args.data_root)
    log.info("found %d hotel files under %s", len(files), os.path.abspath(args.data_root))
    if not files:
        raise SystemExit("no hotel_*_parsed.txt files found")
    raw = [r for f in files for r in parse_file(f)]
    df = filter_reviews(raw)
    ratings, user_map, item_map = index_ids(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(out, index=False)
    user_map.to_csv(out.parent / "user_map.csv", index=False)
    item_map.to_csv(out.parent / "item_map.csv", index=False)
    log.info("wrote %s (%d rows)", out, len(ratings))

    md = stats_markdown(ratings)
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(md)
    print()
    print(md)


if __name__ == "__main__":
    main()
