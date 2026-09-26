"""Real Professional Game Data — Downloads and processes actual game archives.

Replaces fabricated game data with real professional game records from:
- CWI Database: 90,000+ Japanese professional Go games
- JGDB (Joe's Go Database): 500,000+ pro & top amateur games

Usage:
    from training_pipeline.pro_games import build_comprehensive_pro_dataset
    dataset = build_comprehensive_pro_dataset(board_size=9, max_games=1000)
"""

from __future__ import annotations
import os
import glob
from typing import List, Optional
import numpy as np

from gobot_engine.board import Board, Color, Move
from .dataset import GoDataset, GoDataPoint
from .sgf_parser import SGFParser
from .sgf_downloader import download_and_prepare, SGF_SOURCES


def build_comprehensive_pro_dataset(
    board_size: int = 9,
    max_games: Optional[int] = None,
    data_dir: str = "sgf_data",
    source: str = "cwi",
    sgf_dirs: Optional[List[str]] = None,
    min_moves: int = 10,
) -> GoDataset:
    """Builds a GoDataset from REAL professional game archives.

    Downloads actual professional game records (if not already cached),
    parses them, filters by board size, and constructs training positions.

    This replaces the previous approach of using fabricated/synthetic SGF
    strings with actual high-quality game data from public archives.

    Args:
        board_size: Board size to filter for (9, 13, or 19)
        max_games: Maximum number of games to process (None = all)
        data_dir: Directory to cache downloaded SGF archives
        source: Archive source ('cwi' for 90K games, 'jgdb' for 500K games)
        sgf_dirs: If provided, use these directories instead of downloading
        min_moves: Skip games with fewer than this many moves

    Returns:
        GoDataset with training positions from real professional games
    """
    datapoints: List[GoDataPoint] = []

    # Determine SGF file sources
    if sgf_dirs:
        # User-provided directories
        sgf_files = []
        for d in sgf_dirs:
            sgf_files.extend(sorted(glob.glob(os.path.join(d, "**", "*.sgf"), recursive=True)))
        print(f"  Found {len(sgf_files)} SGF files in provided directories")
    else:
        # Download from public archive
        try:
            sgf_files = download_and_prepare(
                source_key=source,
                output_dir=data_dir,
                board_size=board_size,
            )
        except Exception as e:
            print(f"  Warning: Could not download archive: {e}")
            print(f"  Falling back to any local SGF files in {data_dir}/")
            sgf_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.sgf"), recursive=True))

    if not sgf_files:
        print("  No SGF files found. Dataset will be empty.")
        print(f"  Tip: Place .sgf files in '{data_dir}/' or run 'gobot download-games'")
        return GoDataset(datapoints=[], augment_symmetry=True)

    # Parse and process games
    games_accepted = 0
    games_skipped = 0
    games_failed = 0
    total_positions = 0

    print(f"\n  Processing SGF files for {board_size}x{board_size} games...")
    print(f"  Files to scan: {len(sgf_files)}")

    for i, sgf_path in enumerate(sgf_files):
        if max_games is not None and games_accepted >= max_games:
            break

        try:
            games = SGFParser.parse_file(sgf_path)
        except Exception:
            games_failed += 1
            continue

        for game in games:
            if max_games is not None and games_accepted >= max_games:
                break

            # Filter by board size
            if game.size != board_size:
                games_skipped += 1
                continue

            # Filter by minimum move count
            if len(game.moves) < min_moves:
                games_skipped += 1
                continue

            # Must have a winner for meaningful value targets
            if game.winner == Color.EMPTY:
                games_skipped += 1
                continue

            games_accepted += 1

            # Replay game and extract training positions
            positions_in_game = 0
            for board_state, color_to_move, target_move, value_target in SGFParser.replay_game_states(game):
                feat = board_state.to_feature_tensor(color_to_move).numpy()
                act_idx = Move.to_action_index(target_move, board_size)

                datapoints.append(
                    GoDataPoint(
                        feature_tensor=feat,
                        target_action=act_idx,
                        target_value=value_target,
                        board_size=board_size,
                    )
                )
                positions_in_game += 1

            total_positions += positions_in_game

        # Progress reporting
        if (i + 1) % max(1, len(sgf_files) // 20) == 0 or i == len(sgf_files) - 1:
            pct = (i + 1) * 100 // len(sgf_files)
            print(f"  [{pct:3d}%] Scanned {i+1}/{len(sgf_files)} files | "
                  f"Games: {games_accepted} accepted, {games_skipped} skipped | "
                  f"Positions: {total_positions}")

    print(f"\n  ┌─────────────────────────────────────────────────")
    print(f"  │ Dataset Summary")
    print(f"  ├─────────────────────────────────────────────────")
    print(f"  │ Board size:       {board_size}x{board_size}")
    print(f"  │ Games accepted:   {games_accepted}")
    print(f"  │ Games skipped:    {games_skipped}")
    print(f"  │ Parse failures:   {games_failed}")
    print(f"  │ Training positions: {total_positions}")
    print(f"  │ Augmented (8x sym): ~{total_positions * 8}")
    print(f"  └─────────────────────────────────────────────────\n")

    return GoDataset(datapoints=datapoints, augment_symmetry=True)
