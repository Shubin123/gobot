"""Smoke Tests: Core Rules, Board State, GTP Engine, and Neural Network Shapes."""

import pytest
import torch
import numpy as np

from gobot_engine.board import Board, Color, Move, PASS_MOVE, RESIGN_MOVE
from gobot_engine.neural_net import GoResNet, DualHeadLoss
from gobot_engine.gtp import GTPEngine


def test_board_initialization():
    board = Board(size=9, komi=7.5)
    assert board.size == 9
    assert board.komi == 7.5
    assert board.to_move == Color.BLACK
    assert not board.is_game_over
    assert np.all(board.grid == Color.EMPTY)


def test_gtp_coordinates():
    assert Move.to_gtp((0, 0), 19) == "A19"
    assert Move.to_gtp((18, 18), 19) == "T1"
    assert Move.from_gtp("A19", 19) == (0, 0)
    assert Move.from_gtp("T1", 19) == (18, 18)
    assert Move.to_gtp(PASS_MOVE, 19) == "pass"
    assert Move.from_gtp("pass", 19) == PASS_MOVE
    assert Move.from_gtp("resign", 19) == RESIGN_MOVE


def test_stone_capture():
    board = Board(size=9)
    # Place a single white stone at (1, 1)
    board.grid[1, 1] = Color.WHITE

    # Surround with 4 black stones
    board.play((0, 1), Color.BLACK)
    board.play((2, 1), Color.BLACK)
    board.play((1, 0), Color.BLACK)
    # Final capture move
    success = board.play((1, 2), Color.BLACK)

    assert success is True
    assert board.grid[1, 1] == Color.EMPTY  # Captured
    assert board.captures[Color.BLACK] == 1


def test_suicide_rule():
    board = Board(size=9)
    # Surround (0, 0) with white stones at (0, 1) and (1, 0)
    board.play((0, 1), Color.WHITE)
    board.play((1, 0), Color.WHITE)

    # Black attempting to play (0, 0) is suicide without capture -> must be illegal
    assert board.is_legal((0, 0), Color.BLACK) is False
    assert board.play((0, 0), Color.BLACK) is False
    assert board.grid[0, 0] == Color.EMPTY


def test_ko_rule():
    board = Board(size=9)
    # Setup classic ko shape
    # B: (1,0), (0,1), (1,2)
    # W: (1,1), (0,2), (2,2), (1,3)
    board.grid[1, 0] = Color.BLACK
    board.grid[0, 1] = Color.BLACK
    board.grid[2, 1] = Color.BLACK
    board.grid[1, 1] = Color.WHITE

    board.grid[0, 2] = Color.WHITE
    board.grid[2, 2] = Color.WHITE
    board.grid[1, 3] = Color.WHITE

    # Black captures at (1, 2)
    board.to_move = Color.BLACK
    assert board.play((1, 2), Color.BLACK) is True
    assert board.grid[1, 1] == Color.EMPTY
    assert board.ko_point == (1, 1)

    # White cannot immediately recapture at (1, 1)
    assert board.is_legal((1, 1), Color.WHITE) is False
    assert board.play((1, 1), Color.WHITE) is False

    # After a pass or another move, ko restriction is lifted
    board.play(PASS_MOVE, Color.WHITE)
    board.play(PASS_MOVE, Color.BLACK)
    assert board.is_game_over is True


def test_area_scoring():
    board = Board(size=9, komi=7.5)
    # Black occupies top row, White occupies bottom row
    for c in range(9):
        board.grid[0, c] = Color.BLACK
        board.grid[8, c] = Color.WHITE

    score = board.calculate_area_score()
    assert "black_score" in score
    assert "white_score" in score
    assert score["white_score"] >= 7.5


def test_neural_net_forward():
    model = GoResNet(board_size=9, in_channels=8, num_filters=32, num_blocks=2)
    dummy_input = torch.randn(4, 8, 9, 9)
    policy_logits, value = model(dummy_input)

    assert policy_logits.shape == (4, 9 * 9 + 1)
    assert value.shape == (4, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_gtp_engine_commands():
    engine = GTPEngine(board_size=9)
    assert engine.handle_command("protocol_version").strip() == "= 2"
    assert engine.handle_command("name").strip() == "= GoBot-Engine"
    assert engine.handle_command("boardsize 9").strip() == "="
    assert engine.handle_command("komi 6.5").strip() == "="
    assert engine.handle_command("play B E5").strip() == "="
    assert engine.board.grid[4, 4] == Color.BLACK
    gen_resp = engine.handle_command("genmove W").strip()
    assert gen_resp.startswith("=")
