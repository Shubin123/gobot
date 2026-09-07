"""Deterministic Tactical Ladder (Shicho) Reader with Recursive Lookahead."""

from __future__ import annotations
from typing import Tuple, List, Optional, Set
from .board import Board, Color, Move, PASS_MOVE


class LadderSolver:
    """Reads ahead tactical ladder (shicho) sequences to determine if a capture works."""

    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth

    def is_ladder_capturable(
        self,
        board: Board,
        prey_coord: Tuple[int, int],
        attacker_color: int,
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """Simulates whether the prey group can be captured in a ladder.
        Returns:
            (can_capture, sequence_of_moves)
        """
        defender_color = Color.opponent(attacker_color)
        if board.grid[prey_coord[0], prey_coord[1]] != defender_color:
            return False, []

        stones, liberties = board.get_group(prey_coord[0], prey_coord[1])
        if len(liberties) > 2 or len(liberties) == 0:
            return False, []

        sequence: List[Tuple[int, int]] = []
        can_capture = self._search_ladder(
            board.clone(),
            prey_coord,
            attacker_color,
            is_attacker_turn=True,
            depth=0,
            seq=sequence,
        )

        return can_capture, sequence

    def _search_ladder(
        self,
        board: Board,
        prey_coord: Tuple[int, int],
        attacker_color: int,
        is_attacker_turn: bool,
        depth: int,
        seq: List[Tuple[int, int]],
    ) -> bool:
        if depth > self.max_depth:
            return False

        defender_color = Color.opponent(attacker_color)
        stones, liberties = board.get_group(prey_coord[0], prey_coord[1])

        if len(liberties) == 0:
            return True  # Captured

        if len(liberties) >= 3:
            return False  # Escaped

        if is_attacker_turn:
            # Attacker's turn: wants to capture
            if len(liberties) == 1:
                cap_move = next(iter(liberties))
                if board.is_legal(cap_move, attacker_color):
                    seq.append(cap_move)
                    return True
                return False

            # 2 liberties: try each attack move that creates atari
            for attack_move in list(liberties):
                b = board.clone()
                if b.play(attack_move, attacker_color):
                    _, test_libs = b.get_group(prey_coord[0], prey_coord[1])
                    if len(test_libs) == 1:
                        seq.append(attack_move)
                        if self._search_ladder(b, prey_coord, attacker_color, False, depth + 1, seq):
                            return True
                        if seq:
                            seq.pop()
            return False
        else:
            # Defender's turn: forced to play the single liberty in atari
            if len(liberties) == 1:
                def_move = next(iter(liberties))
                if not board.is_legal(def_move, defender_color):
                    return True  # Illegal escape -> captured
                b = board.clone()
                b.play(def_move, defender_color)
                seq.append(def_move)
                res = self._search_ladder(b, def_move, attacker_color, True, depth + 1, seq)
                if not res and seq:
                    seq.pop()
                return res

            return False
