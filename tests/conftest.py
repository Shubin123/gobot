"""Shared pytest fixtures for the GoBot test suite."""

from __future__ import annotations
import os
import tempfile
import pytest
import numpy as np
import torch

from gobot_engine.board import Board, Color, Move, PASS_MOVE
from gobot_engine.neural_net import GoResNet
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from starlette.testclient import TestClient
from online_extension.server import create_app

# Suppress SSL key log env var that causes issues on some systems
os.environ.pop("SSLKEYLOGFILE", None)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_model() -> GoResNet:
    """A tiny 9×9 model with minimal filters — fast for unit tests."""
    return GoResNet(board_size=9, in_channels=8, num_filters=16, num_blocks=1)


@pytest.fixture(scope="session")
def small_model() -> GoResNet:
    """Standard-ish 9×9 model for integration tests."""
    return GoResNet(board_size=9, in_channels=8, num_filters=32, num_blocks=2)


@pytest.fixture(scope="session")
def saved_model_path(small_model, tmp_path_factory) -> str:
    """Save a small model to a temp file and return the path."""
    path = str(tmp_path_factory.mktemp("checkpoints") / "test_model.pt")
    small_model.save_checkpoint(path, extra_meta={"test": True})
    return path


# ---------------------------------------------------------------------------
# Board fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_board_9() -> Board:
    return Board(size=9, komi=7.5)


@pytest.fixture
def empty_board_19() -> Board:
    return Board(size=19, komi=7.5)


@pytest.fixture
def atari_board() -> Board:
    """Single white stone at (2,2) with 3 liberties filled — in atari."""
    board = Board(size=9)
    board.grid[2, 2] = Color.WHITE
    board.grid[1, 2] = Color.BLACK
    board.grid[3, 2] = Color.BLACK
    board.grid[2, 1] = Color.BLACK
    return board


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcts_no_model() -> MCTS:
    return MCTS(num_simulations=10)


@pytest.fixture
def mcts_with_model(tiny_model) -> MCTS:
    return MCTS(model=tiny_model, num_simulations=10)


@pytest.fixture
def optimizer_no_model(mcts_no_model) -> MoveOptimizer:
    return MoveOptimizer(mcts_no_model)


# ---------------------------------------------------------------------------
# API fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def api_client() -> TestClient:
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# SGF fixtures
# ---------------------------------------------------------------------------

SAMPLE_SGF_9x9 = (
    "(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]PB[AlphaGo]PW[Master]BR[9p]WR[9p]RE[B+R]"
    ";B[ee];W[cg];B[eg];W[fc];B[ec];W[eb];B[fd];W[dc];B[ed];W[gc])"
)


@pytest.fixture
def sample_sgf_9x9() -> str:
    return SAMPLE_SGF_9x9


@pytest.fixture
def sample_sgf_file(tmp_path, sample_sgf_9x9) -> str:
    p = tmp_path / "sample.sgf"
    p.write_text(sample_sgf_9x9)
    return str(p)
