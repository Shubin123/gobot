"""Smart Game Format (SGF) Parser and Board State Extractor."""

from __future__ import annotations
import re
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Generator
import numpy as np

from gobot_engine.board import Board, Color, Move, PASS_MOVE


@dataclass
class SGFGame:
    size: int = 19
    komi: float = 7.5
    player_black: str = "Black"
    player_white: str = "White"
    rank_black: str = ""
    rank_white: str = ""
    result: str = ""
    winner: int = Color.EMPTY
    handicap: int = 0
    added_black: List[Tuple[int, int]] = field(default_factory=list)
    added_white: List[Tuple[int, int]] = field(default_factory=list)
    moves: List[Tuple[int, Tuple[int, int]]] = field(default_factory=list)  # (color, move)


class SGFParser:
    """Parses SGF strings and files into SGFGame structures and training tuples."""

    @staticmethod
    def parse_file(file_path: str) -> List[SGFGame]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SGF file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return SGFParser.parse_string(content)

    @staticmethod
    def parse_string(sgf_text: str) -> List[SGFGame]:
        """Parses an SGF string (which may contain multiple games)."""
        games: List[SGFGame] = []
        # Find root game trees enclosed in (...)
        game_trees = re.findall(r"\(\s*;.*?\)(?=\s*\(|\s*$)", sgf_text, flags=re.DOTALL)
        if not game_trees:
            # Fallback if no outer parenthesis
            cleaned = sgf_text.strip()
            if cleaned.startswith("(") and cleaned.endswith(")"):
                game_trees = [cleaned]
            elif ";" in cleaned:
                game_trees = ["(" + cleaned + ")"]

        for tree in game_trees:
            game = SGFParser._parse_single_tree(tree)
            if game and (game.moves or game.added_black or game.added_white):
                games.append(game)

        return games

    @staticmethod
    def _parse_single_tree(tree_text: str) -> Optional[SGFGame]:
        game = SGFGame()

        # Extract root node properties
        # SZ[19]
        sz_match = re.search(r"SZ\[(\d+)\]", tree_text, re.IGNORECASE)
        if sz_match:
            game.size = int(sz_match.group(1))

        # KM[7.5]
        km_match = re.search(r"KM\[([\d\.\-]+)\]", tree_text, re.IGNORECASE)
        if km_match:
            try:
                game.komi = float(km_match.group(1))
            except ValueError:
                game.komi = 7.5

        # PB, PW, BR, WR, RE
        pb_match = re.search(r"PB\[(.*?)\]", tree_text)
        if pb_match:
            game.player_black = pb_match.group(1)
        pw_match = re.search(r"PW\[(.*?)\]", tree_text)
        if pw_match:
            game.player_white = pw_match.group(1)
        br_match = re.search(r"BR\[(.*?)\]", tree_text)
        if br_match:
            game.rank_black = br_match.group(1)
        wr_match = re.search(r"WR\[(.*?)\]", tree_text)
        if wr_match:
            game.rank_white = wr_match.group(1)
        re_match = re.search(r"RE\[(.*?)\]", tree_text)
        if re_match:
            game.result = re_match.group(1).upper()
            if game.result.startswith("B+"):
                game.winner = Color.BLACK
            elif game.result.startswith("W+"):
                game.winner = Color.WHITE

        # Handicap AB[...] / AW[...]
        ab_matches = re.findall(r"AB(?:\[([a-z]{2})\])+", tree_text, re.IGNORECASE)
        for ab in re.finditer(r"AB(\[[a-z]{2}\])+", tree_text, re.IGNORECASE):
            coords = re.findall(r"\[([a-z]{2})\]", ab.group(0), re.IGNORECASE)
            for c in coords:
                game.added_black.append(Move.from_sgf(c))

        for aw in re.finditer(r"AW(\[[a-z]{2}\])+", tree_text, re.IGNORECASE):
            coords = re.findall(r"\[([a-z]{2})\]", aw.group(0), re.IGNORECASE)
            for c in coords:
                game.added_white.append(Move.from_sgf(c))

        # Moves: ;B[cd] or ;W[qp]
        move_matches = re.findall(r";\s*([BW])\[([a-z]{0,2})\]", tree_text, re.IGNORECASE)
        for color_char, coord_str in move_matches:
            color = Color.BLACK if color_char.upper() == "B" else Color.WHITE
            move = Move.from_sgf(coord_str)
            game.moves.append((color, move))

        return game

    @staticmethod
    def replay_game_states(game: SGFGame) -> Generator[Tuple[Board, int, Tuple[int, int], float], None, None]:
        """Replays game move by move, yielding (board_state, color_to_move, target_move, final_game_value).
        Value is +1.0 if color_to_move won, -1.0 if lost, 0.0 if unknown/draw.
        """
        board = Board(size=game.size, komi=game.komi)

        # Apply handicap stones
        for stone in game.added_black:
            board.grid[stone[0], stone[1]] = Color.BLACK
        for stone in game.added_white:
            board.grid[stone[0], stone[1]] = Color.WHITE

        for color, move in game.moves:
            board.to_move = color
            # Value calculation
            if game.winner == color:
                val = 1.0
            elif game.winner == Color.opponent(color):
                val = -1.0
            else:
                val = 0.0

            # Yield current state BEFORE the move is executed
            yield board.clone(), color, move, val

            # Apply move
            if not board.play(move, color):
                # If an illegal move was encountered in SGF (e.g. out of sync), stop replay
                break
