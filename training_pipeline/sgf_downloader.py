"""SGF Game Data Downloader — Fetches real professional game archives for training.

Downloads and processes real Go game records from public archives:
- JGDB (Joe's Go Database): 500K+ pro games for ML training
- CWI Professional Go Database: 90K+ Japanese pro games including 9x9
- GoKifu: Individual pro game downloads

All sources are public domain or freely available for research.
"""

from __future__ import annotations
import os
import sys
import tarfile
import zipfile
import shutil
import glob
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


# =============================================================================
# Public SGF Archive Sources
# =============================================================================
SGF_SOURCES = {
    "cwi": {
        "name": "CWI Professional Go Database",
        "url": "https://homepages.cwi.nl/~aeb/go/games/games.tgz",
        "description": "90,000+ Japanese professional Go games (includes 9x9 directory)",
        "size_mb": 46,
        "format": "tar.gz",
    },
    "jgdb": {
        "name": "Joe's Go Database (JGDB)",
        "url": "https://data.pjreddie.com/files/jgdb.tar.gz",
        "description": "500,000+ pro & top amateur games, optimized for ML training",
        "size_mb": 194,
        "format": "tar.gz",
    },
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Download progress callback."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    else:
        mb_down = downloaded / (1024 * 1024)
        print(f"\r  Downloaded {mb_down:.1f} MB...", end="", flush=True)


def download_archive(
    source_key: str = "cwi",
    output_dir: str = "sgf_data",
    force: bool = False,
) -> str:
    """Downloads an SGF archive from a public source.

    Args:
        source_key: One of 'cwi', 'jgdb'
        output_dir: Directory to store downloaded/extracted files
        force: If True, re-download even if files already exist

    Returns:
        Path to the directory containing extracted SGF files
    """
    if source_key not in SGF_SOURCES:
        raise ValueError(f"Unknown source '{source_key}'. Choose from: {list(SGF_SOURCES.keys())}")

    source = SGF_SOURCES[source_key]
    os.makedirs(output_dir, exist_ok=True)

    extract_dir = os.path.join(output_dir, source_key)
    archive_path = os.path.join(output_dir, f"{source_key}.tar.gz")

    # Check if already extracted
    if os.path.isdir(extract_dir) and not force:
        sgf_count = len(glob.glob(os.path.join(extract_dir, "**", "*.sgf"), recursive=True))
        if sgf_count > 0:
            print(f"  {source['name']} already extracted: {sgf_count} SGF files in {extract_dir}")
            return extract_dir

    # Download
    print(f"\n{'='*60}")
    print(f"  Downloading: {source['name']}")
    print(f"  URL: {source['url']}")
    print(f"  Size: ~{source['size_mb']} MB")
    print(f"{'='*60}\n")

    try:
        urlretrieve(source["url"], archive_path, reporthook=_progress_hook)
        print("\n  Download complete!")
    except (URLError, HTTPError) as e:
        print(f"\n  Download failed: {e}")
        raise RuntimeError(f"Failed to download {source['name']}: {e}") from e

    # Extract
    print(f"  Extracting to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)

    if source["format"] == "tar.gz":
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir, filter="data")
    elif source["format"] == "zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_dir)

    # Count extracted files
    sgf_count = len(glob.glob(os.path.join(extract_dir, "**", "*.sgf"), recursive=True))
    print(f"  Extracted {sgf_count} SGF files.")

    # Clean up archive to save space
    if os.path.exists(archive_path):
        os.remove(archive_path)
        print(f"  Cleaned up archive file.")

    return extract_dir


def collect_sgf_files(
    sgf_dir: str,
    board_size: Optional[int] = None,
    max_files: Optional[int] = None,
) -> List[str]:
    """Collects SGF file paths from a directory tree.

    Args:
        sgf_dir: Root directory to search
        board_size: If specified, filter by board size (checks file path for hints)
        max_files: Maximum number of files to return

    Returns:
        List of absolute paths to SGF files
    """
    all_sgfs = sorted(glob.glob(os.path.join(sgf_dir, "**", "*.sgf"), recursive=True))

    if board_size is not None:
        # Try to filter by directory name hints (e.g., "9x9" directory)
        size_filtered = [f for f in all_sgfs if f"{board_size}x{board_size}" in f]
        if size_filtered:
            all_sgfs = size_filtered
        # Otherwise we'll filter during parsing (by reading SZ[] property)

    if max_files is not None:
        all_sgfs = all_sgfs[:max_files]

    return all_sgfs


def download_and_prepare(
    source_key: str = "cwi",
    output_dir: str = "sgf_data",
    board_size: Optional[int] = None,
    max_files: Optional[int] = None,
    force: bool = False,
) -> List[str]:
    """One-step: download archive + collect SGF files.

    Args:
        source_key: Data source ('cwi' or 'jgdb')
        output_dir: Where to store data
        board_size: Filter for specific board size
        max_files: Limit number of files
        force: Re-download even if exists

    Returns:
        List of SGF file paths ready for training
    """
    extract_dir = download_archive(source_key, output_dir, force)
    files = collect_sgf_files(extract_dir, board_size, max_files)

    print(f"\n  Ready: {len(files)} SGF files for training")
    if board_size:
        print(f"  Board size filter: {board_size}x{board_size}")
    print()

    return files


def list_available_sources() -> None:
    """Prints all available SGF data sources."""
    print(f"\n{'='*60}")
    print(f"  Available Go Game Archives")
    print(f"{'='*60}")
    for key, src in SGF_SOURCES.items():
        print(f"\n  [{key}] {src['name']}")
        print(f"    {src['description']}")
        print(f"    Size: ~{src['size_mb']} MB | Format: {src['format']}")
        print(f"    URL: {src['url']}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    list_available_sources()
    if len(sys.argv) > 1:
        source = sys.argv[1]
        download_and_prepare(source)
    else:
        print("Usage: python -m training_pipeline.sgf_downloader [cwi|jgdb]")
