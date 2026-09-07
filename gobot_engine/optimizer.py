"""Tactical Heuristics, Ladder Solver, Joseki Opening Book, and Move Optimizer."""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set, Any
import numpy as np

from .board import Board, Color, Move, PASS_MOVE, RESIGN_MOVE
from .mcts import MCTS
from .ladder import LadderSolver
from .joseki import JosekiBook


class MoveOptimizer:
    """Combines MCTS search with tactical heuristics, ladder solving, and opening books."""

    def __init__(
        self,
        mcts: MCTS,
        enable_tactics: bool = True,
        use_joseki_book: bool = True,
    ):
        self.mcts = mcts
        self.enable_tactics = enable_tactics
        self.use_joseki_book = use_joseki_book
        self.ladder_solver = LadderSolver(max_depth=50)
        self.joseki_book = JosekiBook()

    def is_true_eye(self, board: Board, r: int, c: int, color: int) -> bool:
        """Determines if (r, c) is a true eye for 'color' (to prevent filling one's own eyes)."""
        if board.grid[r, c] != Color.EMPTY:
            return False

        # All 4 direct cardinal neighbors must be friendly stones or edge
        for ar, ac in board.get_adjacent(r, c):
            if board.grid[ar, ac] != color:
                return False

        # Diagonal neighbors check
        diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        diag_opp_count = 0
        total_diags = 0
        opp = Color.opponent(color)

        for dr, dc in diagonals:
            nr, nc = r + dr, c + dc
            if board.in_bounds(nr, nc):
                total_diags += 1
                if board.grid[nr, nc] == opp:
                    diag_opp_count += 1

        if total_diags < 4:
            return diag_opp_count == 0
        return diag_opp_count <= 1

    def find_immediate_atari_captures(self, board: Board, color: int) -> List[Tuple[int, int]]:
        """Finds opponent groups with exactly 1 liberty (in atari) that can be captured immediately."""
        opp = Color.opponent(color)
        capture_moves: Set[Tuple[int, int]] = set()

        for r in range(board.size):
            for c in range(board.size):
                if board.grid[r, c] == opp:
                    _, liberties = board.get_group(r, c)
                    if len(liberties) == 1:
                        target = next(iter(liberties))
                        if board.is_legal(target, color):
                            capture_moves.add(target)

        return list(capture_moves)

    def find_endangered_group_saves(self, board: Board, color: int) -> List[Tuple[int, int]]:
        """Finds friendly groups in atari (1 liberty) and tests if playing that liberty saves the group."""
        save_moves: Set[Tuple[int, int]] = set()

        for r in range(board.size):
            for c in range(board.size):
                if board.grid[r, c] == color:
                    stones, liberties = board.get_group(r, c)
                    if len(liberties) == 1:
                        target = next(iter(liberties))
                        if board.is_legal(target, color):
                            test_board = board.clone()
                            test_board.play(target, color)
                            test_stones, test_libs = test_board.get_group(target[0], target[1])
                            if len(test_libs) > 1:
                                save_moves.add(target)

        return list(save_moves)

    def is_self_atari_blunder(self, board: Board, move: Tuple[int, int], color: int) -> bool:
        """Checks if a move is a self-atari blunder (giving the opponent a trivial 1-stone capture)."""
        if Move.is_pass(move) or Move.is_resign(move):
            return False

        r, c = move
        test_board = board.clone()
        if not test_board.play(move, color):
            return True

        stones, liberties = test_board.get_group(r, c)
        if len(liberties) == 1:
            opp = Color.opponent(color)
            for ar, ac in test_board.get_adjacent(r, c):
                if test_board.grid[ar, ac] == opp:
                    _, opp_libs = test_board.get_group(ar, ac)
                    if len(opp_libs) == 1:
                        return False
            return True

        return False

    def optimize_move(
        self,
        board: Board,
        temperature: float = 0.0,
        time_limit_sec: float = 5.0,
    ) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        """Selects and optimizes the best move using Joseki book, MCTS search, and tactical refinement."""
        current_color = board.to_move

        # 0. Immediate critical atari capture check
        if self.enable_tactics:
            atari_captures = self.find_immediate_atari_captures(board, current_color)
            if atari_captures:
                cap_move = atari_captures[0]
                return cap_move, {
                    "selected_move": cap_move,
                    "gtp_coord": Move.to_gtp(cap_move, board.size),
                    "winrate": 0.85,
                    "tactical_override": "critical_immediate_atari_capture",
                    "top_candidates": [
                        {
                            "action_index": Move.to_action_index(cap_move, board.size),
                            "move": cap_move,
                            "gtp_coord": Move.to_gtp(cap_move, board.size),
                            "visits": 100,
                            "prior": 0.99,
                            "winrate": 0.85,
                        }
                    ],
                }

        # 1. Opening Joseki Book lookup
        if self.use_joseki_book and len(board.move_history) < 8:
            book_move = self.joseki_book.get_book_move(board)
            if book_move is not None and board.is_legal(book_move, current_color):
                return book_move, {
                    "selected_move": book_move,
                    "gtp_coord": Move.to_gtp(book_move, board.size),
                    "winrate": 0.52,
                    "tactical_override": "joseki_opening_book",
                    "top_candidates": [
                        {
                            "action_index": Move.to_action_index(book_move, board.size),
                            "move": book_move,
                            "gtp_coord": Move.to_gtp(book_move, board.size),
                            "visits": 100,
                            "prior": 0.95,
                            "winrate": 0.52,
                        }
                    ],
                }

        # 2. Run MCTS Search
        candidates = self.mcts.get_top_candidates(board, top_k=5)

        if not candidates:
            return PASS_MOVE, {"reason": "no_legal_moves", "winrate": 0.0, "candidates": []}

        best_cand = candidates[0]
        selected_move = best_cand["move"]
        tactical_override = None

        # 3. Tactical Refinement
        if self.enable_tactics and not Move.is_pass(selected_move):
            atari_captures = self.find_immediate_atari_captures(board, current_color)
            endangered_saves = self.find_endangered_group_saves(board, current_color)

            # Check for immediate critical captures
            if atari_captures and selected_move not in atari_captures:
                selected_move = atari_captures[0]
                tactical_override = "critical_immediate_atari_capture"

            # Avoid filling true eye
            elif self.is_true_eye(board, selected_move[0], selected_move[1], current_color):
                for alt in candidates[1:]:
                    alt_move = alt["move"]
                    if not Move.is_pass(alt_move) and not self.is_true_eye(board, alt_move[0], alt_move[1], current_color):
                        selected_move = alt_move
                        tactical_override = "avoid_filling_true_eye"
                        break

            # Avoid self-atari blunder
            elif self.is_self_atari_blunder(board, selected_move, current_color):
                for alt in candidates[1:]:
                    alt_move = alt["move"]
                    if not self.is_self_atari_blunder(board, alt_move, current_color):
                        selected_move = alt_move
                        tactical_override = "avoid_self_atari_blunder"
                        break

            # Save critical endangered group
            elif endangered_saves and best_cand["visits"] < (self.mcts.num_simulations // 2):
                if endangered_saves[0] not in [c["move"] for c in candidates[:2]]:
                    selected_move = endangered_saves[0]
                    tactical_override = "critical_endangered_group_save"

        metadata = {
            "selected_move": selected_move,
            "gtp_coord": Move.to_gtp(selected_move, board.size),
            "winrate": best_cand["winrate"],
            "tactical_override": tactical_override,
            "top_candidates": candidates,
        }

        return selected_move, metadata
