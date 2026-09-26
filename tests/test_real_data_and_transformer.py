"""Tests for GoTransformerNet, RLTrainer, and Real Professional Data pipeline."""

import os
import torch
import pytest

from gobot_engine.board import Board, Color, Move
from gobot_engine.neural_net import GoTransformerNet, GoResNet, DualHeadLoss
from training_pipeline.pro_games import build_comprehensive_pro_dataset
from training_pipeline.rl_trainer import RLTrainer, ReplayBuffer, EMA
from training_pipeline.dataset import GoDataPoint


def test_transformer_net_forward():
    model = GoTransformerNet(board_size=9, in_channels=8, embed_dim=64, num_heads=2, num_layers=2)
    x = torch.randn(2, 8, 9, 9)
    policy_logits, value = model(x)
    assert policy_logits.shape == (2, 82)
    assert value.shape == (2, 1)
    assert -1.0 <= value.min().item() and value.max().item() <= 1.0


def test_transformer_net_predict_and_checkpoint(tmp_path):
    model = GoTransformerNet(board_size=9, in_channels=8, embed_dim=64, num_heads=2, num_layers=2)
    board = Board(size=9)
    feat = board.to_feature_tensor(Color.BLACK)
    probs, val = model.predict(feat, torch.device("cpu"))
    assert probs.shape == (82,)
    assert abs(probs.sum().item() - 1.0) < 1e-4
    assert -1.0 <= val <= 1.0

    ckpt_path = str(tmp_path / "transformer.pt")
    model.save_checkpoint(ckpt_path)
    loaded = GoTransformerNet.load_checkpoint(ckpt_path, device="cpu")
    assert loaded.board_size == 9
    assert loaded.embed_dim == 64
    assert loaded.num_layers == 2


def test_replay_buffer_and_ema():
    buffer = ReplayBuffer(max_size=100)
    dp = GoDataPoint(
        feature_tensor=torch.zeros(8, 9, 9).numpy(),
        target_action=40,
        target_value=1.0,
        board_size=9,
    )
    buffer.add_game([dp, dp])
    assert len(buffer) == 2
    batch = buffer.sample(1)
    assert len(batch) == 1

    model = GoResNet(board_size=9, num_filters=16, num_blocks=1)
    ema = EMA(model, decay=0.9)
    ema.update(model)
    ema.apply_shadow(model)


def test_pro_dataset_from_local_or_cache():
    # If sgf_data/cwi exists, verify we can load positions
    if os.path.exists("sgf_data/cwi"):
        dataset = build_comprehensive_pro_dataset(board_size=9, max_games=5, data_dir="sgf_data", source="cwi")
        assert len(dataset) > 0
        x, target_policy, target_value = dataset[0]
        assert x.shape == (8, 9, 9)
        assert -1.0 <= target_value.item() <= 1.0
