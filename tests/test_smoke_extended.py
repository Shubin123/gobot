"""
Extended smoke tests — fast sanity checks (< 2s each) for all imports,
CLI argument parsing, server factory, and tensor board representations.

All tests in this file are marked @pytest.mark.smoke.
"""

from __future__ import annotations
import sys
import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_import_gobot_engine():
    from gobot_engine import (
        Board, Color, Move, PASS_MOVE, RESIGN_MOVE,
        GoResNet, DualHeadLoss,
        MCTS, MCTSNode,
        MoveOptimizer,
        GTPEngine,
    )
    assert Board is not None
    assert GoResNet is not None


@pytest.mark.smoke
def test_import_training_pipeline():
    from training_pipeline import (
        SGFParser, GoDataset, GoDataPoint,
        TrainingCurriculum, CurriculumConfig,
        GoTrainer, SelfPlayWorker,
    )
    assert SGFParser is not None


@pytest.mark.smoke
def test_import_online_extension():
    from online_extension.server import create_app
    assert create_app is not None


@pytest.mark.smoke
def test_import_cli():
    import cli
    assert hasattr(cli, "main")


# ---------------------------------------------------------------------------
# Server factory smoke
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_create_app_no_model():
    from online_extension.server import create_app
    app = create_app(model_path=None)
    assert app is not None
    assert app.title == "GoBot Engine & Online Extension API"


@pytest.mark.smoke
def test_create_app_with_missing_model_path():
    """A non-existent model path should not crash server creation."""
    from online_extension.server import create_app
    app = create_app(model_path="/nonexistent/path/model.pt")
    assert app is not None


# ---------------------------------------------------------------------------
# Board representation smoke
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_board_to_tensor_shape():
    from gobot_engine.board import Board, Color
    b = Board(size=9)
    b.play((4, 4), Color.BLACK)
    t = b.to_feature_tensor()
    assert t.shape == (8, 9, 9)
    assert t.dtype == torch.float32


@pytest.mark.smoke
def test_board_render_ascii():
    from gobot_engine.board import Board, Color
    b = Board(size=9)
    b.play((4, 4), Color.BLACK)
    s = b.render_ascii()
    assert isinstance(s, str)
    assert len(s) > 0


@pytest.mark.smoke
def test_board_get_adjacent():
    from gobot_engine.board import Board
    b = Board(size=9)
    adj = b.get_adjacent(0, 0)
    assert len(adj) == 2  # Corner has 2 neighbors
    adj_center = b.get_adjacent(4, 4)
    assert len(adj_center) == 4


# ---------------------------------------------------------------------------
# Neural net smoke
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_model_instantiation_default():
    from gobot_engine.neural_net import GoResNet
    model = GoResNet()
    assert model.board_size == 19
    assert model.num_filters == 64
    assert model.num_blocks == 6


@pytest.mark.smoke
def test_model_parameter_count():
    from gobot_engine.neural_net import GoResNet
    model = GoResNet(board_size=9, num_filters=16, num_blocks=1)
    total_params = sum(p.numel() for p in model.parameters())
    # Should have some parameters, but not astronomical for a tiny model
    assert 1_000 < total_params < 1_000_000


@pytest.mark.smoke
def test_dual_head_loss_instantiation():
    from gobot_engine.neural_net import DualHeadLoss
    loss = DualHeadLoss(value_weight=0.5)
    assert loss.value_weight == 0.5


# ---------------------------------------------------------------------------
# MCTS smoke
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_mcts_default_params():
    from gobot_engine.mcts import MCTS
    mcts = MCTS()
    assert mcts.c_puct == 2.0
    assert mcts.num_simulations == 100


@pytest.mark.smoke
def test_mcts_node_default():
    from gobot_engine.mcts import MCTSNode
    node = MCTSNode(prior=0.5)
    assert node.visit_count == 0
    assert not node.is_expanded


# ---------------------------------------------------------------------------
# CLI arg parser smoke
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_cli_gtp_subcommand_parses(monkeypatch):
    import argparse
    monkeypatch.setattr(sys, "argv", ["gobot", "gtp", "--board-size", "9"])
    from cli import main
    # We just test that argparse doesn't raise — we mock the actual func
    import cli
    original_run = None
    called = []
    def fake_run_gtp(args):
        called.append(args)
    monkeypatch.setattr(cli, "run_gtp", fake_run_gtp)
    monkeypatch.setattr(sys, "argv", ["gobot", "gtp", "--board-size", "9", "--simulations", "5"])
    main()
    assert len(called) == 1
    assert called[0].board_size == 9


@pytest.mark.smoke
def test_cli_server_subcommand_parses(monkeypatch):
    import cli
    called = []
    def fake_run_server(args):
        called.append(args)
    monkeypatch.setattr(cli, "run_server", fake_run_server)
    monkeypatch.setattr(sys, "argv", ["gobot", "server", "--port", "9999"])
    cli.main()
    assert called[0].port == 9999


# ---------------------------------------------------------------------------
# Training pipeline smoke
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_sgf_parser_imports_and_parse_empty():
    from training_pipeline.sgf_parser import SGFParser
    games = SGFParser.parse_string("(;GM[1]SZ[9]KM[7.5])")
    # Empty game (no moves) — should not crash
    assert isinstance(games, list)


@pytest.mark.smoke
def test_dataset_length():
    from training_pipeline.dataset import GoDataset, GoDataPoint
    dps = [GoDataPoint(np.zeros((8, 9, 9), np.float32), 0, 1.0, 9)]
    ds = GoDataset(dps, augment_symmetry=False)
    assert len(ds) == 1


@pytest.mark.smoke
def test_curriculum_config_defaults():
    from training_pipeline.curriculum import CurriculumConfig
    cfg = CurriculumConfig(sources=[], board_size=19)
    assert cfg.board_size == 19
    assert cfg.min_rank == "any"
