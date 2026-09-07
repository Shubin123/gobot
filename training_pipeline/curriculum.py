"""Curriculum Engine: Dictates what, where, and how the Go bot is trained."""

from __future__ import annotations
import os
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

from .sgf_parser import SGFParser, SGFGame
from .dataset import GoDataset, GoDataPoint
from gobot_engine.board import Move


@dataclass
class CurriculumConfig:
    """Configures the training curriculum criteria."""
    sources: List[str] = field(default_factory=list)  # Directories or file patterns
    board_size: int = 19
    min_rank: str = "any"  # "any", "1d", "5d", "pro"
    game_phase: str = "all"  # "all", "opening" (<=30), "middlegame" (31-150), "endgame" (>150)
    winner_perspective_only: bool = False  # If True, only train on the winning player's moves
    max_games: Optional[int] = None
    sample_ratio: float = 1.0


class TrainingCurriculum:
    """Discovers, filters, and prepares datasets according to the configured curriculum."""

    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()

    def _rank_passes_filter(self, rank_str: str, min_rank: str) -> bool:
        if min_rank.lower() == "any" or not min_rank:
            return True
        rank_str = rank_str.strip().lower()
        if not rank_str:
            return False

        if min_rank.lower() == "pro":
            return "p" in rank_str or "pro" in rank_str

        # Parse dan ranks (e.g. "5d", "7d", "9p")
        if "p" in rank_str:
            return True  # Any pro passes dan filter
        if "d" in rank_str:
            try:
                num = int("".join(filter(str.isdigit, rank_str)))
                req_num = int("".join(filter(str.isdigit, min_rank)))
                return num >= req_num
            except ValueError:
                return True
        return False

    def collect_sgf_files(self) -> List[str]:
        files: List[str] = []
        for src in self.config.sources:
            if os.path.isfile(src):
                files.append(src)
            elif os.path.isdir(src):
                found = glob.glob(os.path.join(src, "**", "*.sgf"), recursive=True)
                files.extend(found)
            else:
                # Glob pattern
                found = glob.glob(src, recursive=True)
                files.extend(found)
        return sorted(list(set(files)))

    def build_dataset(self) -> Tuple[GoDataset, Dict[str, Any]]:
        """Scans curriculum sources, applies filters, and builds a GoDataset."""
        sgf_files = self.collect_sgf_files()
        datapoints: List[GoDataPoint] = []
        parsed_games_count = 0
        accepted_games_count = 0
        skipped_games_count = 0

        for file_path in sgf_files:
            try:
                games = SGFParser.parse_file(file_path)
            except Exception:
                continue

            for game in games:
                parsed_games_count += 1
                if self.config.max_games and accepted_games_count >= self.config.max_games:
                    break

                # Size check
                if game.size != self.config.board_size:
                    skipped_games_count += 1
                    continue

                # Rank filter check
                if not (self._rank_passes_filter(game.rank_black, self.config.min_rank) or
                        self._rank_passes_filter(game.rank_white, self.config.min_rank)):
                    skipped_games_count += 1
                    continue

                accepted_games_count += 1

                # Replay moves and construct training points
                move_num = 0
                for board_state, color_to_move, target_move, value_target in SGFParser.replay_game_states(game):
                    move_num += 1

                    # Game phase filter
                    if self.config.game_phase == "opening" and move_num > 30:
                        break
                    elif self.config.game_phase == "middlegame" and (move_num <= 30 or move_num > 150):
                        continue
                    elif self.config.game_phase == "endgame" and move_num <= 150:
                        continue

                    # Winner perspective filter
                    if self.config.winner_perspective_only and game.winner != color_to_move:
                        continue

                    # Sampling ratio
                    if self.config.sample_ratio < 1.0 and np.random.rand() > self.config.sample_ratio:
                        continue

                    feat_tensor = board_state.to_feature_tensor(color_to_move).numpy()
                    action_idx = Move.to_action_index(target_move, game.size)

                    datapoints.append(
                        GoDataPoint(
                            feature_tensor=feat_tensor,
                            target_action=action_idx,
                            target_value=value_target,
                            board_size=game.size,
                        )
                    )

            if self.config.max_games and accepted_games_count >= self.config.max_games:
                break

        dataset = GoDataset(datapoints=datapoints, augment_symmetry=True)

        stats = {
            "sources_scanned": len(self.config.sources),
            "files_found": len(sgf_files),
            "total_games_parsed": parsed_games_count,
            "accepted_games": accepted_games_count,
            "skipped_games": skipped_games_count,
            "total_training_positions": len(datapoints),
            "board_size": self.config.board_size,
            "curriculum_filter": {
                "min_rank": self.config.min_rank,
                "game_phase": self.config.game_phase,
                "winner_perspective_only": self.config.winner_perspective_only,
            },
        }

        return dataset, stats
