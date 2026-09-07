"""
End-to-End Game Tests — full scenarios from first move to game-over.

Covers:
- Complete human-vs-bot terminal game (7×7, limited moves)
- Bot-vs-bot self-play that generates SGF output
- Full API game via HTTP: reset → play → bot move → resign
- ONNX export and reload (inference equivalence)
- WebSocket game flow via Starlette test client
"""

from __future__ import annotations
import os
import json
import struct
import tempfile
import pytest
import numpy as np
import torch

from gobot_engine.board import Board, Color, Move, PASS_MOVE, RESIGN_MOVE
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from gobot_engine.neural_net import GoResNet
from training_pipeline.trainer import GoTrainer
from training_pipeline.self_play import SelfPlayWorker
from online_extension.server import create_app
from starlette.testclient import TestClient


# ===========================================================================
# Bot-vs-Bot game playthrough
# ===========================================================================

@pytest.mark.e2e
def test_e2e_bot_vs_bot_7x7_completes():
    """Two MCTS bots play a full 7×7 game. Game must end and have a result."""
    bot_b = MCTS(num_simulations=8)
    bot_w = MCTS(num_simulations=8)
    result = GoTrainer.play_match(bot_b, bot_w, board_size=7, komi=5.5, max_moves=30)

    assert "winner" in result
    assert "result_str" in result
    assert result["moves_played"] > 0
    # winner is a Color int (1=BLACK, 2=WHITE) or string label
    assert result["winner"] in ("B", "W", "draw", 1, 2, 0, None) or isinstance(result["winner"], int)


@pytest.mark.e2e
def test_e2e_bot_vs_bot_with_model(small_model):
    """Bot-vs-bot where both bots use a neural model."""
    bot_b = MCTS(model=small_model, num_simulations=8)
    bot_w = MCTS(model=small_model, num_simulations=8)
    result = GoTrainer.play_match(bot_b, bot_w, board_size=7, komi=5.5, max_moves=25)
    assert "result_str" in result


# ===========================================================================
# Human-vs-bot simulation (scripted moves)
# ===========================================================================

@pytest.mark.e2e
def test_e2e_human_vs_bot_scripted():
    """Simulate a human playing scripted moves against the bot on 7×7."""
    board = Board(size=7, komi=5.5)
    mcts = MCTS(num_simulations=8)
    optimizer = MoveOptimizer(mcts, enable_tactics=False, use_joseki_book=False)

    # Human (Black) makes 5 scripted moves; bot (White) responds each time
    human_moves = [(3, 3), (2, 2), (4, 4), (1, 1), (5, 5)]
    moves_played = 0

    for human_move in human_moves:
        if board.is_game_over:
            break
        # Human plays
        if board.is_legal(human_move, Color.BLACK):
            board.play(human_move, Color.BLACK)
            moves_played += 1

        if board.is_game_over:
            break

        # Bot responds
        bot_move, meta = optimizer.optimize_move(board)
        board.play(bot_move, Color.WHITE)
        moves_played += 1

    assert moves_played > 0
    score = board.calculate_area_score()
    assert "result_str" in score


# ===========================================================================
# Self-play with SGF output
# ===========================================================================

@pytest.mark.e2e
def test_e2e_self_play_sgf_generation(tmp_path):
    """Self-play should produce valid SGF files and training data."""
    worker = SelfPlayWorker(board_size=7, num_simulations=5)
    dataset = worker.generate_batch(
        num_games=2, max_moves_per_game=15,
        sgf_output_dir=str(tmp_path),
    )
    assert len(dataset) > 0

    sgf_files = list(tmp_path.glob("*.sgf"))
    assert len(sgf_files) >= 1

    # Each SGF should be parseable
    from training_pipeline.sgf_parser import SGFParser
    for sgf_file in sgf_files:
        content = sgf_file.read_text()
        games = SGFParser.parse_string(content)
        assert len(games) >= 1


# ===========================================================================
# Full API game (HTTP)
# ===========================================================================

@pytest.mark.e2e
def test_e2e_api_full_game_with_resign():
    """Full API game: reset → play → bot move → pass × 2 → game over."""
    app = create_app()
    client = TestClient(app)

    # 1. Health check
    r = client.get("/api/health")
    assert r.status_code == 200

    # 2. Reset to 9×9
    r = client.post("/api/board/reset", json={"board_size": 9, "komi": 7.5})
    assert r.status_code == 200

    # 3. Play opening moves
    moves = [("B", "E5"), ("W", "D4"), ("B", "F6"), ("W", "C3")]
    for color, coord in moves:
        r = client.post("/api/board/play", json={"color": color, "coord": coord})
        assert r.status_code == 200

    # 4. Request bot move
    r = client.post("/api/optimize/move", json={"num_simulations": 15})
    assert r.status_code == 200
    data = r.json()
    assert "selected_move" in data
    assert "top_candidates" in data
    assert len(data["top_candidates"]) > 0

    # 5. Analyze position
    r = client.post("/api/analyze", json={"num_simulations": 15})
    assert r.status_code == 200
    analysis = r.json()
    assert "candidates" in analysis
    assert "current_score_estimate" in analysis

    # 6. Resign
    state = client.get("/api/board/state").json()
    resign_color = state["to_move"]
    r = client.post("/api/board/play", json={"color": resign_color, "coord": "resign"})
    assert r.status_code == 200

    # 7. Verify game is over
    r = client.get("/api/board/state")
    assert r.json()["is_game_over"] is True


@pytest.mark.e2e
def test_e2e_api_board_size_transitions():
    """Switching board sizes via reset should work cleanly."""
    app = create_app()
    client = TestClient(app)

    for size in [9, 13, 19]:
        r = client.post("/api/board/reset", json={"board_size": size, "komi": 7.5})
        assert r.status_code == 200
        assert r.json()["board_size"] == size

        state = client.get("/api/board/state")
        assert state.json()["board_size"] == size
        grid = state.json()["grid"]
        assert len(grid) == size
        assert len(grid[0]) == size


# ===========================================================================
# WebSocket game flow
# ===========================================================================

@pytest.mark.e2e
def test_e2e_websocket_get_state():
    """WebSocket /ws/game should return board state on get_state action."""
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"action": "get_state"})
        data = ws.receive_json()
        assert "board_size" in data
        assert "grid" in data
        assert "to_move" in data


@pytest.mark.e2e
def test_e2e_websocket_play_move():
    """WebSocket play action should update board state."""
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"action": "play", "color": "B", "coord": "E5"})
        data = ws.receive_json()
        assert data["to_move"] == "W"  # Turn switched

        ws.send_json({"action": "get_state"})
        state = ws.receive_json()
        # E5 on 19×19 board is row=14, col=4
        board_size = state["board_size"]
        grid = state["grid"]
        r = board_size - 5
        c = 4  # E = index 4
        assert grid[r][c] == Color.BLACK


@pytest.mark.e2e
def test_e2e_websocket_genmove():
    """WebSocket genmove should return bot_move field."""
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"action": "genmove", "color": "B"})
        data = ws.receive_json()
        assert "bot_move" in data
        assert "winrate" in data


# ===========================================================================
# ONNX export and reload
# ===========================================================================

@pytest.mark.e2e
def test_e2e_onnx_export_and_inference(tmp_path, small_model):
    """Export model to ONNX, reload with onnxruntime, verify inference matches."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    onnx_path = str(tmp_path / "model.onnx")

    # Export
    small_model.eval()
    dummy = torch.randn(1, 8, 9, 9)
    torch.onnx.export(
        small_model,
        dummy,
        onnx_path,
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={"board": {0: "batch"}},
        opset_version=17,
    )

    assert os.path.exists(onnx_path)

    # Reload with ONNX Runtime
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_outputs = sess.run(None, {inp_name: dummy.numpy()})

    policy_ort = ort_outputs[0]  # (1, 82)
    value_ort = ort_outputs[1]   # (1, 1)

    # Compare with PyTorch output
    with torch.no_grad():
        policy_pt, value_pt = small_model(dummy)

    assert np.allclose(policy_ort, policy_pt.numpy(), atol=1e-4)
    assert np.allclose(value_ort, value_pt.numpy(), atol=1e-4)


@pytest.mark.e2e
def test_e2e_onnx_shard_and_reassemble(tmp_path, small_model):
    """Shard the ONNX model bytes into chunks and reassemble correctly."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    onnx_path = str(tmp_path / "model.onnx")
    dummy = torch.randn(1, 8, 9, 9)
    small_model.eval()
    torch.onnx.export(
        small_model, dummy, onnx_path,
        input_names=["board"], output_names=["policy_logits", "value"],
        dynamic_axes={"board": {0: "batch"}}, opset_version=17,
    )

    # Shard into 3 pieces
    with open(onnx_path, "rb") as f:
        model_bytes = f.read()

    total_size = len(model_bytes)
    num_shards = 3
    shard_size = (total_size + num_shards - 1) // num_shards
    shard_paths = []
    manifest = {"version": "1.0", "shards": [], "total_size": total_size}

    for i in range(num_shards):
        chunk = model_bytes[i * shard_size: (i + 1) * shard_size]
        shard_path = str(tmp_path / f"model.shard{i:03d}.bin")
        with open(shard_path, "wb") as f:
            f.write(chunk)
        shard_paths.append(shard_path)
        manifest["shards"].append({
            "index": i,
            "filename": f"model.shard{i:03d}.bin",
            "size": len(chunk),
        })

    manifest_path = str(tmp_path / "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    # Reassemble
    reassembled = b""
    for sp in shard_paths:
        with open(sp, "rb") as f:
            reassembled += f.read()

    assert reassembled == model_bytes

    # Verify reassembled model still works
    reassembled_path = str(tmp_path / "reassembled.onnx")
    with open(reassembled_path, "wb") as f:
        f.write(reassembled)

    sess = ort.InferenceSession(reassembled_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})
    assert ort_out[0].shape == (1, 9 * 9 + 1)
