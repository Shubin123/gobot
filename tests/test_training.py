"""Training Pipeline, SGF Parsing, Curriculum and Trainer Tests."""

import os
import tempfile
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from gobot_engine.board import Board, Color, Move
from gobot_engine.neural_net import GoResNet
from training_pipeline.sgf_parser import SGFParser, SGFGame
from training_pipeline.dataset import GoDataset, GoDataPoint
from training_pipeline.curriculum import TrainingCurriculum, CurriculumConfig
from training_pipeline.trainer import GoTrainer


SAMPLE_SGF = """(;GM[1]FF[4]CA[UTF-8]AP[Sabaki:0.51.1]SZ[9]KM[7.5]RU[Chinese]PB[AlphaGo]PW[Master]BR[9p]WR[9p]RE[B+R]
;B[ee];W[cg];B[eg];W[fc];B[ec];W[eb];B[fd];W[dc];B[ed];W[gc];B[cc];W[cd];B[dd];W[cb];B[bc];W[bb];B[df];W[ce];B[cf];W[bf];B[dg])"""


def test_sgf_parser():
    games = SGFParser.parse_string(SAMPLE_SGF)
    assert len(games) == 1
    game = games[0]
    assert game.size == 9
    assert game.komi == 7.5
    assert game.player_black == "AlphaGo"
    assert game.player_white == "Master"
    assert game.rank_black == "9p"
    assert game.winner == Color.BLACK
    assert len(game.moves) == 21
    assert game.moves[0] == (Color.BLACK, (4, 4))  # ee -> (4,4)


def test_dataset_dihedral_augmentation():
    feat = np.zeros((8, 9, 9), dtype=np.float32)
    feat[0, 2, 3] = 1.0
    dp = GoDataPoint(feature_tensor=feat, target_action=2 * 9 + 3, target_value=1.0, board_size=9)

    dataset = GoDataset([dp], augment_symmetry=True)
    x, y_p, y_v = dataset[0]

    assert x.shape == (8, 9, 9)
    assert y_v.item() == 1.0
    assert 0 <= y_p.item() < 9 * 9


def test_curriculum_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        sgf_file = os.path.join(tmpdir, "game.sgf")
        with open(sgf_file, "w") as f:
            f.write(SAMPLE_SGF)

        # 1. Matching curriculum
        config_ok = CurriculumConfig(
            sources=[tmpdir],
            board_size=9,
            min_rank="5d",
            game_phase="opening",
        )
        curr = TrainingCurriculum(config_ok)
        ds, stats = curr.build_dataset()
        assert stats["accepted_games"] == 1
        assert len(ds) > 0

        # 2. Filter out non-matching board size
        config_skip = CurriculumConfig(sources=[tmpdir], board_size=19)
        curr_skip = TrainingCurriculum(config_skip)
        ds_skip, stats_skip = curr_skip.build_dataset()
        assert stats_skip["accepted_games"] == 0
        assert len(ds_skip) == 0


def test_trainer_fit_loop():
    # Construct small synthetic dataset
    datapoints = []
    for _ in range(30):
        feat = np.random.randn(8, 9, 9).astype(np.float32)
        act = np.random.randint(0, 82)
        val = float(np.random.choice([-1.0, 1.0]))
        datapoints.append(GoDataPoint(feat, act, val, board_size=9))

    dataset = GoDataset(datapoints, augment_symmetry=False)
    model = GoResNet(board_size=9, in_channels=8, num_filters=16, num_blocks=1)
    trainer = GoTrainer(model, lr=0.01)

    with tempfile.TemporaryDirectory() as chk_dir:
        history = trainer.fit(
            dataset,
            epochs=2,
            batch_size=8,
            val_split=0.2,
            checkpoint_dir=chk_dir,
            model_name="test_model.pt",
        )
        assert len(history) == 2
        assert os.path.exists(os.path.join(chk_dir, "test_model.pt"))
