"""End-to-End System, API Server, and Self-Play Integration Tests."""

import os
import tempfile
import pytest

# Ensure SSLKEYLOGFILE does not cause permission issues
os.environ.pop("SSLKEYLOGFILE", None)

from starlette.testclient import TestClient

from gobot_engine.board import Board, Color, Move
from gobot_engine.mcts import MCTS
from training_pipeline.self_play import SelfPlayWorker
from training_pipeline.trainer import GoTrainer
from online_extension.server import create_app


def test_e2e_mcts_game_playout():
    bot_b = MCTS(num_simulations=10)
    bot_w = MCTS(num_simulations=10)

    result = GoTrainer.play_match(bot_b, bot_w, board_size=7, komi=5.5, max_moves=20)
    assert "winner" in result
    assert "result_str" in result
    assert result["moves_played"] > 0


def test_e2e_self_play_generation():
    worker = SelfPlayWorker(board_size=7, num_simulations=10)
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = worker.generate_batch(num_games=1, max_moves_per_game=15, sgf_output_dir=tmpdir)
        assert len(dataset) > 0


def test_e2e_fastapi_server_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # 2. Reset board to 9x9
    res = client.post("/api/board/reset", json={"board_size": 9, "komi": 7.5})
    assert res.status_code == 200
    assert res.json()["board_size"] == 9

    # 3. Play a move
    res = client.post("/api/board/play", json={"color": "B", "coord": "E5"})
    assert res.status_code == 200
    assert res.json()["to_move"] == "W"

    # 4. Request optimal move
    res = client.post("/api/optimize/move", json={"num_simulations": 30, "enable_tactics": True})
    assert res.status_code == 200
    data = res.json()
    assert "selected_move" in data
    assert "winrate" in data
    assert len(data["top_candidates"]) > 0

    # 5. Analyze position
    res = client.post("/api/analyze", json={"num_simulations": 30})
    assert res.status_code == 200
    assert "candidates" in res.json()
