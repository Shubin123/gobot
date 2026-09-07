"""Capability and Tactical Optimization Tests."""

import pytest
import numpy as np

from gobot_engine.board import Board, Color, Move, PASS_MOVE
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from gobot_engine.neural_net import GoResNet


def test_immediate_atari_capture():
    board = Board(size=9)
    # White stone at (2, 2) has 3 liberties filled by black, 1 remaining at (2, 3)
    board.grid[2, 2] = Color.WHITE
    board.grid[1, 2] = Color.BLACK
    board.grid[3, 2] = Color.BLACK
    board.grid[2, 1] = Color.BLACK

    mcts = MCTS(num_simulations=30)
    optimizer = MoveOptimizer(mcts)

    captures = optimizer.find_immediate_atari_captures(board, Color.BLACK)
    assert (2, 3) in captures

    # Move optimizer should prioritize the capture
    move, meta = optimizer.optimize_move(board)
    assert move == (2, 3)


def test_endangered_group_save():
    board = Board(size=9)
    # Black stone at (4, 4) in atari (only liberty at (4, 5))
    board.grid[4, 4] = Color.BLACK
    board.grid[3, 4] = Color.WHITE
    board.grid[5, 4] = Color.WHITE
    board.grid[4, 3] = Color.WHITE

    mcts = MCTS(num_simulations=30)
    optimizer = MoveOptimizer(mcts)

    saves = optimizer.find_endangered_group_saves(board, Color.BLACK)
    assert (4, 5) in saves


def test_true_eye_protection():
    board = Board(size=9)
    # Solid black eye in the top-left corner at (0, 0)
    board.grid[0, 1] = Color.BLACK
    board.grid[1, 0] = Color.BLACK
    board.grid[1, 1] = Color.BLACK

    mcts = MCTS(num_simulations=30)
    optimizer = MoveOptimizer(mcts)

    assert optimizer.is_true_eye(board, 0, 0, Color.BLACK) is True
    # False eye if opponent controls diagonal on corner
    board.grid[1, 1] = Color.WHITE
    assert optimizer.is_true_eye(board, 0, 0, Color.BLACK) is False


def test_mcts_candidate_ranking():
    board = Board(size=9)
    board.play((4, 4), Color.BLACK)

    mcts = MCTS(num_simulations=50)
    candidates = mcts.get_top_candidates(board, top_k=3)

    assert len(candidates) == 3
    assert candidates[0]["visits"] >= candidates[1]["visits"]
    assert "winrate" in candidates[0]
    assert "gtp_coord" in candidates[0]


def test_ladder_solver():
    from gobot_engine.ladder import LadderSolver
    board = Board(size=9)
    board.grid[1, 1] = Color.WHITE
    board.grid[0, 1] = Color.BLACK
    board.grid[1, 0] = Color.BLACK
    board.grid[0, 2] = Color.BLACK
    board.grid[2, 0] = Color.BLACK

    solver = LadderSolver()
    can_capture, seq = solver.is_ladder_capturable(board, (1, 1), Color.BLACK)
    assert can_capture is True
    assert len(seq) > 0


def test_joseki_book_lookup():
    from gobot_engine.joseki import JosekiBook
    book = JosekiBook()

    # Move 0 on 9x9 -> Valid pro opening candidate
    b9 = Board(size=9)
    mv9 = book.get_book_move(b9)
    assert mv9 in [(4, 4), (2, 4), (4, 2), (2, 2), (6, 2), (6, 6), (2, 6), (3, 3), (5, 5)]

    # Move 0 on 19x19 -> Valid star / corner candidate
    b19 = Board(size=19)
    mv19 = book.get_book_move(b19)
    assert mv19 in [(3, 15), (15, 3), (3, 3), (15, 15), (3, 16), (15, 16)]
