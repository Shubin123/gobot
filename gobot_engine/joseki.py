"""High-Dan Opening Joseki and Fuseki Pattern Dictionary."""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
from .board import Board, Color, Move, PASS_MOVE


class JosekiBook:
    """Provides high-winrate opening moves across standard board sizes."""

    def __init__(self):
        # Dictionary format: board_size -> List of move sequence tuples [(r, c), ...]
        self.openings_9x9: List[List[Tuple[int, int]]] = [
            # Tengen (Center) openings
            [(4, 4), (2, 4), (4, 2), (2, 2), (5, 3)],  # E5, E7, C5, C7, D4
            [(4, 4), (2, 6), (6, 2), (2, 2), (6, 6)],  # E5, G7, C3, C7, G3
            [(4, 4), (3, 2), (5, 6), (2, 4), (6, 4)],  # E5, C6, G4, E7, E3
            # Komoku / 3-4 style
            [(2, 4), (4, 4), (6, 4), (2, 2), (5, 3)],  # E7, E5, E3, C7, D4
            [(2, 2), (4, 4), (6, 6), (2, 6), (6, 2)],  # C7, E5, G3, G7, C3
            [(3, 3), (4, 4), (5, 5), (3, 5), (5, 3)],  # D6, E5, F4, D4, F6
        ]

        self.openings_13x13: List[List[Tuple[int, int]]] = [
            # 4-4 Star Points & Center
            [(3, 3), (9, 9), (3, 9), (9, 3), (6, 6)],  # D10, K4, K10, D4, G7
            [(3, 3), (9, 9), (2, 9), (3, 8), (8, 3)],
            [(6, 6), (3, 3), (9, 9), (3, 9), (9, 3)],  # Tengen start
        ]

        self.openings_19x19: List[List[Tuple[int, int]]] = [
            # Modern Pro 4-4 & 3-4 Fuseki
            [(3, 15), (15, 3), (3, 3), (15, 15), (2, 13), (3, 13)],  # Q16, D4, D16, Q4, N17, N16
            [(3, 15), (15, 3), (15, 15), (3, 3), (2, 14), (3, 14)],  # Q16, D4, Q4, D16
            [(3, 15), (15, 3), (3, 3), (15, 15), (2, 2), (3, 2)],    # Early 3-3 invasion pattern
            [(3, 16), (15, 3), (3, 3), (15, 15), (4, 16), (5, 15)],  # 3-4 Komoku approach
            [(3, 15), (15, 3), (4, 2), (14, 16), (2, 4), (16, 14)],  # Cross-opening (diagonal fuseki)
            [(3, 15), (15, 15), (3, 3), (15, 3), (5, 14), (3, 13)],  # Parallel fuseki
        ]

    def get_book_move(self, board: Board) -> Optional[Tuple[int, int]]:
        """Checks if current move history matches any opening book sequences with dihedral symmetry."""
        size = board.size
        history = [move for _, move in board.move_history]
        move_idx = len(history)

        if move_idx >= 8:
            return None  # Beyond opening book phase

        if size == 9:
            candidates_list = self.openings_9x9
        elif size == 13:
            candidates_list = self.openings_13x13
        elif size == 19:
            candidates_list = self.openings_19x19
        else:
            return None

        matching_next_moves: List[Tuple[int, int]] = []

        # Check all openings and all 8 symmetries
        for seq in candidates_list:
            if len(seq) <= move_idx:
                continue

            for rot_k in range(4):
                for do_flip in [False, True]:
                    # Transform sequence
                    trans_seq = [self._transform_coord(m, size, rot_k, do_flip) for m in seq]
                    # Check if prefix matches history
                    if trans_seq[:move_idx] == history:
                        next_mv = trans_seq[move_idx]
                        if board.is_legal(next_mv, board.to_move):
                            matching_next_moves.append(next_mv)

        if matching_next_moves:
            # Pick from valid book variations
            return random.choice(matching_next_moves)

        # Fallback if at start of game (move 0)
        if move_idx == 0:
            if size == 9:
                return (4, 4)  # Tengen on 9x9
            elif size == 13:
                return (3, 3)  # D10 star point on 13x13
            elif size == 19:
                return (3, 15)  # Q16 star point on 19x19

        return None

    @staticmethod
    def _transform_coord(coord: Tuple[int, int], size: int, rot_k: int, do_flip: bool) -> Tuple[int, int]:
        r, c = coord
        grid = np.zeros((size, size), dtype=np.int32)
        grid[r, c] = 1
        grid = np.rot90(grid, k=rot_k)
        if do_flip:
            grid = np.fliplr(grid)
        new_r, new_c = np.argwhere(grid == 1)[0]
        return int(new_r), int(new_c)
