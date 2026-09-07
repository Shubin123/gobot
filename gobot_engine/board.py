"""Go Board State Representation and Rule Enforcement."""

from __future__ import annotations
import copy
from typing import List, Tuple, Set, Optional, Dict, Union
import numpy as np
import torch


class Color:
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @staticmethod
    def opponent(color: int) -> int:
        if color == Color.BLACK:
            return Color.WHITE
        if color == Color.WHITE:
            return Color.BLACK
        return Color.EMPTY

    @staticmethod
    def to_char(color: int) -> str:
        if color == Color.BLACK:
            return "B"
        if color == Color.WHITE:
            return "W"
        return "."


# Special moves
PASS_MOVE: Tuple[int, int] = (-1, -1)
RESIGN_MOVE: Tuple[int, int] = (-2, -2)

# GTP column letters (omits 'I')
GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


class Move:
    @staticmethod
    def is_pass(move: Tuple[int, int]) -> bool:
        return move == PASS_MOVE or move[0] < 0

    @staticmethod
    def is_resign(move: Tuple[int, int]) -> bool:
        return move == RESIGN_MOVE

    @staticmethod
    def to_gtp(move: Tuple[int, int], size: int) -> str:
        if Move.is_pass(move):
            return "pass"
        if Move.is_resign(move):
            return "resign"
        r, c = move
        if not (0 <= r < size and 0 <= c < size):
            return "pass"
        col_char = GTP_COLUMNS[c]
        row_num = size - r
        return f"{col_char}{row_num}"

    @staticmethod
    def from_gtp(coord: str, size: int) -> Tuple[int, int]:
        coord = coord.strip().lower()
        if coord in ("pass", ""):
            return PASS_MOVE
        if coord == "resign":
            return RESIGN_MOVE
        col_char = coord[0].upper()
        if col_char not in GTP_COLUMNS:
            raise ValueError(f"Invalid GTP column: {col_char}")
        c = GTP_COLUMNS.index(col_char)
        try:
            row_num = int(coord[1:])
        except ValueError:
            raise ValueError(f"Invalid GTP row: {coord[1:]}")
        r = size - row_num
        if not (0 <= r < size and 0 <= c < size):
            raise ValueError(f"Coordinates out of bounds: {coord} for board size {size}")
        return (r, c)

    @staticmethod
    def to_sgf(move: Tuple[int, int]) -> str:
        if Move.is_pass(move) or Move.is_resign(move):
            return ""
        r, c = move
        return chr(ord("a") + c) + chr(ord("a") + r)

    @staticmethod
    def from_sgf(coord: str) -> Tuple[int, int]:
        coord = coord.strip().lower()
        if not coord or len(coord) < 2 or coord in ("tt", "pass"):
            return PASS_MOVE
        c = ord(coord[0]) - ord("a")
        r = ord(coord[1]) - ord("a")
        return (r, c)

    @staticmethod
    def to_action_index(move: Tuple[int, int], size: int) -> int:
        """Flattens (r, c) to index 0..(size*size - 1), and PASS is size*size."""
        if Move.is_pass(move) or Move.is_resign(move):
            return size * size
        return move[0] * size + move[1]

    @staticmethod
    def from_action_index(idx: int, size: int) -> Tuple[int, int]:
        if idx >= size * size or idx < 0:
            return PASS_MOVE
        return (idx // size, idx % size)


class Board:
    """Full Go board state with rule enforcement (Suicide prohibition, Ko, Chinese area scoring)."""

    def __init__(self, size: int = 19, komi: float = 7.5):
        self.size: int = size
        self.komi: float = komi
        self.grid: np.ndarray = np.zeros((size, size), dtype=np.int8)
        self.to_move: int = Color.BLACK
        self.ko_point: Optional[Tuple[int, int]] = None
        self.captures: Dict[int, int] = {Color.BLACK: 0, Color.WHITE: 0}
        self.move_history: List[Tuple[int, Tuple[int, int]]] = []  # (color, move)
        self.pass_count: int = 0
        self.is_game_over: bool = False
        self.resigned_by: Optional[int] = None
        self.position_hashes: Set[int] = set()
        self._record_hash()

    def _get_hash(self) -> int:
        return hash((self.grid.tobytes(), self.to_move, self.ko_point))

    def _record_hash(self) -> None:
        self.position_hashes.add(self._get_hash())

    def clone(self) -> Board:
        b = Board(size=self.size, komi=self.komi)
        b.grid = np.copy(self.grid)
        b.to_move = self.to_move
        b.ko_point = self.ko_point
        b.captures = dict(self.captures)
        b.move_history = list(self.move_history)
        b.pass_count = self.pass_count
        b.is_game_over = self.is_game_over
        b.resigned_by = self.resigned_by
        b.position_hashes = set(self.position_hashes)
        return b

    def reset(self) -> None:
        self.grid.fill(Color.EMPTY)
        self.to_move = Color.BLACK
        self.ko_point = None
        self.captures = {Color.BLACK: 0, Color.WHITE: 0}
        self.move_history.clear()
        self.pass_count = 0
        self.is_game_over = False
        self.resigned_by = None
        self.position_hashes.clear()
        self._record_hash()

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def get_adjacent(self, r: int, c: int) -> List[Tuple[int, int]]:
        adj = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                adj.append((nr, nc))
        return adj

    def get_group(self, r: int, c: int) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Returns (stones_in_group, liberties_set)."""
        color = self.grid[r, c]
        if color == Color.EMPTY:
            return set(), set()

        stones: Set[Tuple[int, int]] = {(r, c)}
        liberties: Set[Tuple[int, int]] = set()
        queue = [(r, c)]
        visited = {(r, c)}

        while queue:
            curr_r, curr_c = queue.pop(0)
            for adj_r, adj_c in self.get_adjacent(curr_r, curr_c):
                adj_color = self.grid[adj_r, adj_c]
                if adj_color == Color.EMPTY:
                    liberties.add((adj_r, adj_c))
                elif adj_color == color and (adj_r, adj_c) not in visited:
                    visited.add((adj_r, adj_c))
                    stones.add((adj_r, adj_c))
                    queue.append((adj_r, adj_c))

        return stones, liberties

    def count_liberties(self, r: int, c: int) -> int:
        _, liberties = self.get_group(r, c)
        return len(liberties)

    def is_legal(self, move: Tuple[int, int], color: Optional[int] = None) -> bool:
        if color is None:
            color = self.to_move

        if self.is_game_over:
            return False

        if Move.is_pass(move) or Move.is_resign(move):
            return True

        r, c = move
        if not self.in_bounds(r, c) or self.grid[r, c] != Color.EMPTY:
            return False

        # Ko point prohibition
        if self.ko_point == (r, c):
            return False

        opp = Color.opponent(color)
        # Check if placement has direct liberties
        has_liberty = False
        adjacent = self.get_adjacent(r, c)
        for ar, ac in adjacent:
            if self.grid[ar, ac] == Color.EMPTY:
                has_liberty = True
                break

        # Check if placement captures any opponent stones
        captured_any = False
        # Place stone provisionally
        self.grid[r, c] = color
        for ar, ac in adjacent:
            if self.grid[ar, ac] == opp:
                _, opp_libs = self.get_group(ar, ac)
                if len(opp_libs) == 0:
                    captured_any = True
                    break

        if not has_liberty and not captured_any:
            # Check if joining a friendly group with remaining liberties
            _, friendly_libs = self.get_group(r, c)
            if len(friendly_libs) == 0:
                # Suicide move - illegal
                self.grid[r, c] = Color.EMPTY
                return False

        # Revert provisional placement
        self.grid[r, c] = Color.EMPTY
        return True

    def get_legal_moves(self, color: Optional[int] = None) -> List[Tuple[int, int]]:
        if color is None:
            color = self.to_move

        if self.is_game_over:
            return []

        moves: List[Tuple[int, int]] = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal((r, c), color):
                    moves.append((r, c))
        moves.append(PASS_MOVE)
        return moves

    def play(self, move: Tuple[int, int], color: Optional[int] = None) -> bool:
        """Applies move. Returns True if move was legal and applied, False otherwise."""
        if color is None:
            color = self.to_move

        if not self.is_legal(move, color):
            return False

        if Move.is_resign(move):
            self.is_game_over = True
            self.resigned_by = color
            self.move_history.append((color, move))
            return True

        if Move.is_pass(move):
            self.ko_point = None
            self.pass_count += 1
            self.move_history.append((color, move))
            if self.pass_count >= 2:
                self.is_game_over = True
            self.to_move = Color.opponent(color)
            self._record_hash()
            return True

        self.pass_count = 0
        r, c = move
        opp = Color.opponent(color)
        self.grid[r, c] = color

        # Process captures
        total_captured: List[Tuple[int, int]] = []
        for ar, ac in self.get_adjacent(r, c):
            if self.grid[ar, ac] == opp:
                opp_stones, opp_libs = self.get_group(ar, ac)
                if len(opp_libs) == 0:
                    for sr, sc in opp_stones:
                        self.grid[sr, sc] = Color.EMPTY
                        total_captured.append((sr, sc))

        self.captures[color] += len(total_captured)

        # Check ko creation: exactly 1 stone captured and single stone placed has 1 liberty
        if len(total_captured) == 1:
            placed_stones, placed_libs = self.get_group(r, c)
            if len(placed_stones) == 1 and len(placed_libs) == 1:
                self.ko_point = total_captured[0]
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        self.move_history.append((color, move))
        self.to_move = Color.opponent(color)
        self._record_hash()
        return True

    def calculate_area_score(self) -> Dict[str, Union[float, str]]:
        """Calculates score using Tromp-Taylor / Chinese area rules."""
        black_score = 0.0
        white_score = self.komi

        visited = set()
        territory_grid = np.zeros((self.size, self.size), dtype=np.int8)

        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == Color.BLACK:
                    black_score += 1.0
                    territory_grid[r, c] = Color.BLACK
                elif self.grid[r, c] == Color.WHITE:
                    white_score += 1.0
                    territory_grid[r, c] = Color.WHITE
                elif (r, c) not in visited:
                    # Flood fill empty area
                    empty_region = []
                    borders: Set[int] = set()
                    queue = [(r, c)]
                    visited.add((r, c))

                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        empty_region.append((curr_r, curr_c))
                        for ar, ac in self.get_adjacent(curr_r, curr_c):
                            adj_val = self.grid[ar, ac]
                            if adj_val == Color.EMPTY:
                                if (ar, ac) not in visited:
                                    visited.add((ar, ac))
                                    queue.append((ar, ac))
                            else:
                                borders.add(adj_val)

                    if len(borders) == 1:
                        owner = next(iter(borders))
                        if owner == Color.BLACK:
                            black_score += len(empty_region)
                            for er, ec in empty_region:
                                territory_grid[er, ec] = Color.BLACK
                        elif owner == Color.WHITE:
                            white_score += len(empty_region)
                            for er, ec in empty_region:
                                territory_grid[er, ec] = Color.WHITE

        margin = black_score - white_score
        winner = Color.BLACK if margin > 0 else Color.WHITE
        winner_str = f"B+{margin:.1f}" if margin > 0 else f"W+{-margin:.1f}"
        if self.resigned_by == Color.BLACK:
            winner_str = "W+Resign"
            winner = Color.WHITE
        elif self.resigned_by == Color.WHITE:
            winner_str = "B+Resign"
            winner = Color.BLACK

        return {
            "black_score": black_score,
            "white_score": white_score,
            "margin": margin,
            "winner": winner,
            "result_str": winner_str,
            "territory_grid": territory_grid,
        }

    def to_feature_tensor(self, color: Optional[int] = None) -> torch.Tensor:
        """Converts current board state into an 8-plane spatial tensor for PyTorch.
        Planes:
        0: Current player stones
        1: Opponent stones
        2: Current player 1 liberty
        3: Current player 2 liberties
        4: Opponent 1 liberty
        5: Opponent 2 liberties
        6: Ko point location
        7: Color to move (1 if Black, 0 if White)
        """
        if color is None:
            color = self.to_move
        opp = Color.opponent(color)

        planes = np.zeros((8, self.size, self.size), dtype=np.float32)

        # Stone placements
        planes[0] = (self.grid == color).astype(np.float32)
        planes[1] = (self.grid == opp).astype(np.float32)

        # Liberties
        for r in range(self.size):
            for c in range(self.size):
                cell = self.grid[r, c]
                if cell == color:
                    libs = self.count_liberties(r, c)
                    if libs == 1:
                        planes[2, r, c] = 1.0
                    elif libs == 2:
                        planes[3, r, c] = 1.0
                elif cell == opp:
                    libs = self.count_liberties(r, c)
                    if libs == 1:
                        planes[4, r, c] = 1.0
                    elif libs == 2:
                        planes[5, r, c] = 1.0

        # Ko point
        if self.ko_point is not None:
            kr, kc = self.ko_point
            if self.in_bounds(kr, kc):
                planes[6, kr, kc] = 1.0

        # Player to move
        if color == Color.BLACK:
            planes[7].fill(1.0)
        else:
            planes[7].fill(0.0)

        return torch.from_numpy(planes)

    def render_ascii(self) -> str:
        header = "   " + " ".join(GTP_COLUMNS[: self.size]) + "\n"
        lines = [header]
        for r in range(self.size):
            row_num = self.size - r
            row_str = f"{row_num:2d} "
            for c in range(self.size):
                val = self.grid[r, c]
                ch = Color.to_char(val)
                if (r, c) == self.ko_point:
                    ch = "x"
                row_str += ch + " "
            row_str += f"{row_num:2d}\n"
            lines.append(row_str)
        lines.append(header)
        lines.append(f"Turn: {Color.to_char(self.to_move)} | Captures B:{self.captures[Color.BLACK]} W:{self.captures[Color.WHITE]}\n")
        return "".join(lines)
