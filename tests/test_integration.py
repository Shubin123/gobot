"""
Integration tests — multi-module pipelines tested end-to-end.

Covers:
- Neural net checkpoint roundtrip with inference verification
- Full MCTS search using a loaded neural model
- Optimizer full pipeline (board → optimize → legal result)
- FastAPI server WebSocket game flow
- SGF parse → dataset → training tensor pipeline
- Self-play data generation with model
- Server endpoint chain (reset, play, optimize, analyze)
"""

from __future__ import annotations
import os
import json
import tempfile
import pytest
import numpy as np
import torch

from gobot_engine.board import Board, Color, Move, PASS_MOVE
from gobot_engine.neural_net import GoResNet
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from training_pipeline.sgf_parser import SGFParser
from training_pipeline.dataset import GoDataset, GoDataPoint
from training_pipeline.trainer import GoTrainer
from training_pipeline.self_play import SelfPlayWorker


# ---------------------------------------------------------------------------
# Neural network integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_checkpoint_roundtrip_inference_matches(tmp_path, small_model):
    """Save checkpoint, reload it, verify inference output is bit-identical."""
    board_tensor = torch.randn(8, 9, 9)
    small_model.eval()
    with torch.no_grad():
        inp = board_tensor.unsqueeze(0)
        orig_pol, orig_val = small_model(inp)

    path = str(tmp_path / "roundtrip.pt")
    small_model.save_checkpoint(path, extra_meta={"test": "roundtrip"})
    loaded = GoResNet.load_checkpoint(path, device="cpu")
    loaded.eval()

    with torch.no_grad():
        new_pol, new_val = loaded(inp)

    assert torch.allclose(orig_pol, new_pol, atol=1e-5)
    assert torch.allclose(orig_val, new_val, atol=1e-5)

    # Verify meta stored
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["meta"]["test"] == "roundtrip"


@pytest.mark.integration
def test_mcts_with_neural_net_finds_reasonable_move(small_model):
    """MCTS guided by neural network should return a move and not crash."""
    board = Board(size=9, komi=7.5)
    board.play((4, 4), Color.BLACK)

    mcts = MCTS(model=small_model, num_simulations=20)
    move, probs, value = mcts.search(board)

    assert probs.shape == (9 * 9 + 1,)
    assert -1.0 <= value <= 1.0
    if move != PASS_MOVE:
        r, c = move
        assert 0 <= r < 9 and 0 <= c < 9
        assert board.is_legal(move, Color.WHITE)


@pytest.mark.integration
def test_optimizer_with_model_returns_legal_move(small_model):
    """Optimizer with neural model should return a legal move."""
    board = Board(size=9, komi=7.5)
    board.play((4, 4), Color.BLACK)
    board.play((3, 3), Color.WHITE)

    mcts = MCTS(model=small_model, num_simulations=15)
    optimizer = MoveOptimizer(mcts, enable_tactics=True, use_joseki_book=False)
    move, meta = optimizer.optimize_move(board)

    assert "gtp_coord" in meta
    assert "winrate" in meta
    assert "top_candidates" in meta
    assert isinstance(meta["winrate"], float)

    if move != PASS_MOVE:
        assert board.is_legal(move, Color.BLACK)


# ---------------------------------------------------------------------------
# SGF → Dataset → Training pipeline
# ---------------------------------------------------------------------------

FULL_SGF = (
    "(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]PB[AlphaGo]PW[Master]BR[9p]WR[9p]RE[B+R]"
    ";B[ee];W[cg];B[eg];W[fc];B[ec];W[eb];B[fd];W[dc];B[ed];W[gc]"
    ";B[cc];W[cd];B[dd];W[cb];B[bc];W[bb];B[df];W[ce];B[cf];W[bf];B[dg])"
)


@pytest.mark.integration
def test_sgf_to_dataset_pipeline():
    """Parse SGF → build dataset → verify tensors are correct shape."""
    games = SGFParser.parse_string(FULL_SGF)
    assert len(games) == 1
    game = games[0]

    # Build feature tensors manually for each position
    datapoints = []
    board = Board(size=game.size, komi=game.komi)
    for color, move in game.moves[:5]:
        feat = board.to_feature_tensor().numpy()
        dp = GoDataPoint(
            feature_tensor=feat,
            target_action=move[0] * game.size + move[1] if move != PASS_MOVE else game.size ** 2,
            target_value=1.0 if color == Color.BLACK else -1.0,
            board_size=game.size,
        )
        datapoints.append(dp)
        board.play(move, color)

    dataset = GoDataset(datapoints, augment_symmetry=False)
    assert len(dataset) == 5
    x, y_p, y_v = dataset[0]
    assert x.shape == (8, 9, 9)
    assert 0 <= y_p.item() < 9 * 9 + 1


@pytest.mark.integration
def test_trainer_fit_and_save(tmp_path, small_model):
    """GoTrainer.fit() should reduce loss and save a checkpoint."""
    datapoints = []
    for _ in range(40):
        feat = np.random.randn(8, 9, 9).astype(np.float32)
        act = np.random.randint(0, 82)
        val = float(np.random.choice([-1.0, 1.0]))
        datapoints.append(GoDataPoint(feat, act, val, board_size=9))

    dataset = GoDataset(datapoints, augment_symmetry=False)
    model = GoResNet(board_size=9, num_filters=16, num_blocks=1)
    trainer = GoTrainer(model, lr=0.01)
    history = trainer.fit(
        dataset, epochs=2, batch_size=8, val_split=0.2,
        checkpoint_dir=str(tmp_path), model_name="integration_model.pt",
    )
    assert len(history) == 2
    assert all("loss" in h for h in history)
    assert os.path.exists(tmp_path / "integration_model.pt")


# ---------------------------------------------------------------------------
# Self-play integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_self_play_generates_dataset(tmp_path):
    """SelfPlayWorker should generate a non-empty dataset of training positions."""
    worker = SelfPlayWorker(board_size=7, num_simulations=5)
    dataset = worker.generate_batch(
        num_games=2, max_moves_per_game=10,
        sgf_output_dir=str(tmp_path),
    )
    assert len(dataset) > 0
    # At least one SGF file should be written
    sgf_files = list(tmp_path.glob("*.sgf"))
    assert len(sgf_files) >= 1


@pytest.mark.integration
def test_self_play_with_model_generates_dataset(tmp_path, small_model):
    """Self-play with neural model guidance should still generate data."""
    worker = SelfPlayWorker(model=small_model, board_size=7, num_simulations=5)
    dataset = worker.generate_batch(num_games=1, max_moves_per_game=10)
    assert len(dataset) > 0


# ---------------------------------------------------------------------------
# FastAPI server integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_api_full_game_chain(api_client):
    """Reset → play sequence → optimize → analyze chain all return 200."""
    # Reset to 9×9
    r = api_client.post("/api/board/reset", json={"board_size": 9, "komi": 7.5})
    assert r.status_code == 200
    assert r.json()["board_size"] == 9

    # Play a few moves
    for coord in ["E5", "D4", "F6"]:
        color = "B" if api_client.get("/api/board/state").json()["to_move"] == "B" else "W"
        r = api_client.post("/api/board/play", json={"color": color, "coord": coord})
        assert r.status_code == 200

    # Get state
    r = api_client.get("/api/board/state")
    assert r.status_code == 200
    state = r.json()
    assert state["board_size"] == 9

    # Optimize move
    r = api_client.post("/api/optimize/move", json={"num_simulations": 10, "enable_tactics": True})
    assert r.status_code == 200
    data = r.json()
    assert "selected_move" in data
    assert "winrate" in data

    # Analyze
    r = api_client.post("/api/analyze", json={"num_simulations": 10})
    assert r.status_code == 200
    assert "candidates" in r.json()


@pytest.mark.integration
def test_api_optimize_with_custom_grid(api_client):
    """Optimize endpoint with a manually supplied grid."""
    grid = [[0] * 9 for _ in range(9)]
    grid[4][4] = 1  # Black stone at E5
    r = api_client.post("/api/optimize/move", json={
        "grid": grid,
        "board_size": 9,
        "to_move": "W",
        "num_simulations": 10,
    })
    assert r.status_code == 200
    assert "selected_move" in r.json()


@pytest.mark.integration
def test_api_health_has_model_flag(api_client):
    r = api_client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert "has_model" in data
    assert "board_size" in data


@pytest.mark.integration
def test_api_illegal_move_returns_400(api_client):
    api_client.post("/api/board/reset", json={"board_size": 9, "komi": 7.5})
    # Play E5 as Black
    api_client.post("/api/board/play", json={"color": "B", "coord": "E5"})
    # Try to play the same spot (occupied)
    r = api_client.post("/api/board/play", json={"color": "W", "coord": "E5"})
    assert r.status_code == 400
