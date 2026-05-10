"""
Script to download all required datasets.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --wiki-only
    python scripts/download_data.py --hotpotqa-only
"""

import argparse
import sys
import os

# Allow importing src/ from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import download_wiki_vote, download_hotpotqa
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Download GraphRAG project datasets")
    parser.add_argument("--wiki-only", action="store_true",
                        help="Download only the Wikipedia Vote Network")
    parser.add_argument("--hotpotqa-only", action="store_true",
                        help="Download only the HotpotQA dataset")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Directory to save raw data files (default: data/raw)")
    args = parser.parse_args()

    logger = setup_logging()
    download_wiki = not args.hotpotqa_only
    download_hotpot = not args.wiki_only

    if download_wiki:
        logger.info("Downloading Wikipedia Vote Network ...")
        path = download_wiki_vote(raw_dir=args.raw_dir)
        logger.info(f"Wiki-Vote ready at: {path}")

    if download_hotpot:
        logger.info("Downloading HotpotQA distractor dev set ...")
        path = download_hotpotqa(raw_dir=args.raw_dir)
        logger.info(f"HotpotQA ready at: {path}")

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
