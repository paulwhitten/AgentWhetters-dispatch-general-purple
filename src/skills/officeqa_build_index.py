"""Pre-build the OfficeQA BM25 index and serialize to disk.

Run this script during Docker image build to avoid the 2-5 minute
startup cost of downloading, chunking, and indexing 697 Treasury
Bulletin files at runtime.

Usage:
    python -m skills.officeqa_build_index [--output /path/to/index.pkl.gz]
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import logging
import pickle
import time

from skills.officeqa_rag import _ensure_index, INDEX_PKL

logging.basicConfig(
    level=logging.INFO, format="%(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


async def build() -> None:
    t0 = time.monotonic()
    index = await _ensure_index()
    elapsed_build = time.monotonic() - t0
    logger.info("Index built in %.1fs", elapsed_build)

    t1 = time.monotonic()
    # Drop bm25.corpus -- only needed during __init__ to build tf/idf.
    # Saves ~460 MB in the pickle.
    index.bm25.corpus = []
    data = {
        "chunks": index.chunks,
        "bm25": index.bm25,
        "table_catalog": index.table_catalog,
    }
    INDEX_PKL.parent.mkdir(parents=True, exist_ok=True)
    raw = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    with open(INDEX_PKL, "wb") as f:
        f.write(gzip.compress(raw, compresslevel=1))
    elapsed_save = time.monotonic() - t1

    size_mb = INDEX_PKL.stat().st_size / 1_048_576
    logger.info(
        "Saved index to %s (%.1f MB) in %.1fs",
        INDEX_PKL, size_mb, elapsed_save,
    )
    logger.info(
        "Stats: %d chunks, %d table sections",
        len(index.chunks), len(index.table_catalog),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-build OfficeQA index")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output path (default: INDEX_PKL constant)",
    )
    args = parser.parse_args()
    if args.output:
        import skills.officeqa_rag as mod
        from pathlib import Path
        mod.INDEX_PKL = Path(args.output)
    asyncio.run(build())


if __name__ == "__main__":
    main()
