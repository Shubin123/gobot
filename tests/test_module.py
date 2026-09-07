"""
Module-level unit tests for each public API in GoBot.

Tests individual classes and functions in isolation, covering:
- Board rules (captures, Ko, superko, scoring, legality)
- MCTS internals (node expansion, UCB, value backup)
- Neural network shapes, loss, and predict API
- GTP engine commands
- MoveOptimizer tactical heuristics
"""

from __future__ import annotations
import math
import pytest
import numpy as np
import torch
import torch.nn.functional as F

from gobot_engine.board import Board, Color, Move, PASS_MOVE, RESIGN_MOVE, GTP_COLUMNS
from gobot_engine.neural_net import GoResNet, DualHeadLoss
from gobot_engine.mcts import MCTS, MCTSNode
from gobot_engine.gtp import GTPEngine
from gobot_engine.optimizer import MoveOptimizer
from gobot_engine.ladder import LadderSolver
from gobot_engine.joseki import JosekiBook


# ===========================================================================
# Board
# ===========================================================================

class TestBoardInit:
    @pytest.mark.smoke
    def test_default_9x9(self, empty_board_9):
        b = empty_board_9
        assert b.size == 9
        assert b.komi == 7.5
        assert b.to_move == Color.BLACK
        assert not b.is_game_over
        assert np.all(b.grid == Color.EMPTY)

    @pytest.mark.smoke
    @pytest.mark.parametrize("size", [9, 13, 19])
    def test_various_sizes(self, size):
        b = Board(size=size)
        assert b.grid.shape == (size, size)

    def test_captures_initialized_zero(self, empty_board_9):
        assert empty_board_9.captures[Color.BLACK] == 0
        assert empty_board_9.captures[Color.WHITE] == 0


class TestBoardPlay:
    @pytest.mark.smoke
    def test_basic_play_alternates_turn(self, empty_board_9):
        b = empty_board_9
        assert b.to_move == Color.BLACK
        b.play((4, 4), Color.BLACK)
        assert b.to_move == Color.WHITE

    def test_play_records_history(self, empty_board_9):
        b = empty_board_9
        b.play((4, 4), Color.BLACK)
        assert len(b.move_history) == 1

    def test_play_occupied_is_illegal(self, empty_board_9):
        b = empty_board_9
        b.play((4, 4), Color.BLACK)
        assert b.play((4, 4), Color.WHITE) is False

    def test_capture_single_stone(self, empty_board_9):
        b = empty_board_9
        b.grid[1, 1] = Color.WHITE
        b.play((0, 1), Color.BLACK)
        b.play((2, 1), Color.BLACK)
        b.play((1, 0), Color.BLACK)
        success = b.play((1, 2), Color.BLACK)
        assert success is True
        assert b.grid[1, 1] == Color.EMPTY
        assert b.captures[Color.BLACK] == 1

    def test_capture_group(self):
        b = Board(size=9)
        # Two white stones at (1,1),(1,2) to be captured
        b.grid[1, 1] = Color.WHITE
        b.grid[1, 2] = Color.WHITE
        # Surround them — leave (1,3) as the last liberty
        b.grid[0, 1] = Color.BLACK
        b.grid[0, 2] = Color.BLACK
        b.grid[1, 0] = Color.BLACK
        b.grid[2, 1] = Color.BLACK
        b.grid[2, 2] = Color.BLACK
        b.to_move = Color.BLACK
        # Capture by playing the last liberty
        result = b.play((1, 3), Color.BLACK)
        assert result is True
        assert b.grid[1, 1] == Color.EMPTY
        assert b.grid[1, 2] == Color.EMPTY
        assert b.captures[Color.BLACK] == 2

    def test_suicide_is_illegal(self):
        b = Board(size=9)
        b.play((0, 1), Color.WHITE)
        b.play((1, 0), Color.WHITE)
        assert b.is_legal((0, 0), Color.BLACK) is False
        assert b.play((0, 0), Color.BLACK) is False

    def test_ko_prevents_immediate_recapture(self):
        b = Board(size=9)
        b.grid[1, 0] = Color.BLACK
        b.grid[0, 1] = Color.BLACK
        b.grid[2, 1] = Color.BLACK
        b.grid[1, 1] = Color.WHITE
        b.grid[0, 2] = Color.WHITE
        b.grid[2, 2] = Color.WHITE
        b.grid[1, 3] = Color.WHITE
        b.to_move = Color.BLACK
        assert b.play((1, 2), Color.BLACK) is True
        assert b.ko_point == (1, 1)
        assert b.is_legal((1, 1), Color.WHITE) is False

    def test_ko_released_after_pass(self):
        b = Board(size=9)
        b.grid[1, 0] = Color.BLACK
        b.grid[0, 1] = Color.BLACK
        b.grid[2, 1] = Color.BLACK
        b.grid[1, 1] = Color.WHITE
        b.grid[0, 2] = Color.WHITE
        b.grid[2, 2] = Color.WHITE
        b.grid[1, 3] = Color.WHITE
        b.to_move = Color.BLACK
        b.play((1, 2), Color.BLACK)
        b.play(PASS_MOVE, Color.WHITE)
        assert b.is_legal((1, 1), Color.WHITE) is True

    def test_double_pass_ends_game(self, empty_board_9):
        b = empty_board_9
        b.play(PASS_MOVE, Color.BLACK)
        b.play(PASS_MOVE, Color.WHITE)
        assert b.is_game_over is True

    def test_resign_ends_game(self, empty_board_9):
        b = empty_board_9
        b.play(RESIGN_MOVE, Color.BLACK)
        assert b.is_game_over is True


class TestBoardScoring:
    def test_area_score_structure(self, empty_board_9):
        score = empty_board_9.calculate_area_score()
        assert "black_score" in score
        assert "white_score" in score
        assert "result_str" in score

    def test_komi_applied_to_white(self):
        b = Board(size=9, komi=7.5)
        score = b.calculate_area_score()
        # Empty board: white gets komi, so white wins
        assert score["white_score"] == pytest.approx(7.5, abs=0.1)

    def test_black_territory_counted(self):
        b = Board(size=9, komi=0.0)
        for c in range(9):
            b.grid[0, c] = Color.BLACK
        score = b.calculate_area_score()
        assert score["black_score"] > score["white_score"]


class TestGTPCoordinates:
    @pytest.mark.smoke
    @pytest.mark.parametrize("move,size,expected", [
        ((0, 0), 19, "A19"),
        ((18, 18), 19, "T1"),
        ((4, 4), 9, "E5"),
        (PASS_MOVE, 9, "pass"),
    ])
    def test_to_gtp(self, move, size, expected):
        assert Move.to_gtp(move, size) == expected

    @pytest.mark.smoke
    def test_resign_detected(self):
        assert Move.is_resign(RESIGN_MOVE) is True
        assert Move.is_pass(PASS_MOVE) is True
        assert not Move.is_resign(PASS_MOVE)

    @pytest.mark.smoke
    @pytest.mark.parametrize("coord,size,expected", [
        ("A19", 19, (0, 0)),
        ("T1", 19, (18, 18)),
        ("E5", 9, (4, 4)),
        ("pass", 9, PASS_MOVE),
        ("resign", 9, RESIGN_MOVE),
    ])
    def test_from_gtp(self, coord, size, expected):
        assert Move.from_gtp(coord, size) == expected

    def test_gtp_no_i_column(self):
        """GTP skips the letter 'I'."""
        assert "I" not in GTP_COLUMNS

    def test_invalid_gtp_raises(self):
        with pytest.raises(ValueError):
            Move.from_gtp("Z99", 9)  # Out of bounds


class TestBoardClone:
    def test_clone_is_independent(self, empty_board_9):
        b = empty_board_9
        b.play((4, 4), Color.BLACK)
        clone = b.clone()
        clone.play((3, 3), Color.WHITE)
        # original unaffected
        assert b.grid[3, 3] == Color.EMPTY
        assert clone.grid[3, 3] == Color.WHITE


# ===========================================================================
# Neural Network
# ===========================================================================

class TestNeuralNet:
    @pytest.mark.smoke
    @pytest.mark.parametrize("board_size,filters,blocks", [
        (9, 16, 1),
        (9, 32, 2),
        (19, 64, 4),
    ])
    def test_forward_shapes(self, board_size, filters, blocks):
        model = GoResNet(board_size=board_size, num_filters=filters, num_blocks=blocks)
        x = torch.randn(2, 8, board_size, board_size)
        policy, value = model(x)
        assert policy.shape == (2, board_size * board_size + 1)
        assert value.shape == (2, 1)

    @pytest.mark.smoke
    def test_value_in_range(self, tiny_model):
        x = torch.randn(4, 8, 9, 9)
        _, value = tiny_model(x)
        assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

    def test_policy_softmax_sums_to_one(self, tiny_model):
        x = torch.randn(4, 8, 9, 9)
        logits, _ = tiny_model(x)
        probs = F.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_predict_api_single_board(self, tiny_model):
        board_tensor = torch.randn(8, 9, 9)
        probs, val = tiny_model.predict(board_tensor)
        assert probs.shape == (9 * 9 + 1,)
        assert -1.0 <= val <= 1.0
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_predict_is_deterministic_in_eval(self, tiny_model):
        tiny_model.eval()
        board_tensor = torch.randn(8, 9, 9)
        p1, v1 = tiny_model.predict(board_tensor)
        p2, v2 = tiny_model.predict(board_tensor)
        assert torch.allclose(p1, p2)
        assert abs(v1 - v2) < 1e-6

    def test_checkpoint_roundtrip(self, tiny_model, tmp_path):
        path = str(tmp_path / "model.pt")
        tiny_model.save_checkpoint(path, extra_meta={"epoch": 1})
        loaded = GoResNet.load_checkpoint(path)
        # Verify parameters are identical
        for (n1, p1), (n2, p2) in zip(tiny_model.named_parameters(), loaded.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_dual_head_loss_soft_targets(self, tiny_model):
        loss_fn = DualHeadLoss(value_weight=1.0)
        B, A = 4, 9 * 9 + 1
        pred_policy = torch.randn(B, A)
        pred_value = torch.tanh(torch.randn(B, 1))
        target_policy = F.softmax(torch.randn(B, A), dim=1)
        target_value = torch.FloatTensor(B, 1).uniform_(-1, 1)
        total, pol_loss, val_loss = loss_fn(pred_policy, pred_value, target_policy, target_value)
        assert total.item() > 0
        assert pol_loss.item() >= 0
        assert val_loss.item() >= 0
        assert abs(total.item() - (pol_loss + val_loss).item()) < 1e-5

    def test_dual_head_loss_hard_targets(self):
        loss_fn = DualHeadLoss()
        B, A = 4, 82
        pred_policy = torch.randn(B, A)
        pred_value = torch.tanh(torch.randn(B, 1))
        target_policy = torch.randint(0, A, (B,))
        target_value = torch.FloatTensor(B, 1).uniform_(-1, 1)
        total, _, _ = loss_fn(pred_policy, pred_value, target_policy, target_value)
        assert total.item() > 0


# ===========================================================================
# MCTS Node
# ===========================================================================

class TestMCTSNode:
    def test_q_value_no_visits(self):
        node = MCTSNode(prior=0.5)
        assert node.q_value == 0.0

    def test_q_value_with_visits(self):
        node = MCTSNode(prior=0.5)
        node.visit_count = 4
        node.value_sum = 2.0
        assert node.q_value == pytest.approx(0.5)

    def test_ucb_increases_with_parent_visits(self):
        node = MCTSNode(prior=0.3)
        ucb_low = node.ucb_score(10, c_puct=2.0)
        ucb_high = node.ucb_score(100, c_puct=2.0)
        assert ucb_high > ucb_low

    def test_ucb_decreases_with_child_visits(self):
        node = MCTSNode(prior=0.3)
        node.visit_count = 0
        ucb_0 = node.ucb_score(100, c_puct=2.0)
        node.visit_count = 10
        ucb_10 = node.ucb_score(100, c_puct=2.0)
        assert ucb_0 > ucb_10


class TestMCTS:
    @pytest.mark.smoke
    def test_search_returns_legal_move(self, empty_board_9, mcts_no_model):
        move, probs, value = mcts_no_model.search(empty_board_9)
        # Move must be legal on an empty 9x9 board
        if move != PASS_MOVE:
            r, c = move
            assert 0 <= r < 9 and 0 <= c < 9
        assert probs.shape == (9 * 9 + 1,)
        assert -1.0 <= value <= 1.0

    def test_search_with_neural_model(self, empty_board_9, mcts_with_model):
        move, probs, value = mcts_with_model.search(empty_board_9)
        assert probs.shape == (9 * 9 + 1,)

    def test_get_top_candidates(self, empty_board_9, mcts_no_model):
        mcts_no_model.num_simulations = 20
        candidates = mcts_no_model.get_top_candidates(empty_board_9, top_k=3)
        assert len(candidates) <= 3
        for c in candidates:
            assert "gtp_coord" in c
            assert "visits" in c
            assert "winrate" in c

    def test_dirichlet_noise_applied(self, empty_board_9, mcts_no_model):
        # Should not raise; confirms noise path executes
        move, _, _ = mcts_no_model.search(empty_board_9, add_noise=True)
        assert move is not None


# ===========================================================================
# GTP Engine
# ===========================================================================

class TestGTPEngine:
    @pytest.fixture
    def engine(self):
        return GTPEngine(board_size=9)

    @pytest.mark.smoke
    def test_protocol_version(self, engine):
        assert engine.handle_command("protocol_version").strip() == "= 2"

    @pytest.mark.smoke
    def test_name(self, engine):
        assert engine.handle_command("name").strip() == "= GoBot-Engine"

    def test_boardsize(self, engine):
        resp = engine.handle_command("boardsize 9")
        assert resp.strip() == "="

    def test_komi(self, engine):
        resp = engine.handle_command("komi 6.5")
        assert resp.strip() == "="
        assert engine.board.komi == 6.5

    def test_play_black(self, engine):
        resp = engine.handle_command("play B E5")
        assert resp.strip() == "="
        assert engine.board.grid[4, 4] == Color.BLACK

    def test_play_white(self, engine):
        engine.handle_command("play B E5")
        resp = engine.handle_command("play W D4")
        assert resp.strip() == "="

    def test_genmove(self, engine):
        resp = engine.handle_command("genmove B").strip()
        assert resp.startswith("=")

    def test_clear_board(self, engine):
        engine.handle_command("play B E5")
        engine.handle_command("clear_board")
        assert np.all(engine.board.grid == Color.EMPTY)

    def test_known_command(self, engine):
        resp = engine.handle_command("known_command play").strip()
        assert "true" in resp.lower()

    def test_unknown_command(self, engine):
        resp = engine.handle_command("totally_unknown_cmd")
        assert "?" in resp

    def test_list_commands(self, engine):
        resp = engine.handle_command("list_commands")
        assert "genmove" in resp
        assert "play" in resp

    def test_final_score(self, engine):
        resp = engine.handle_command("final_score")
        assert resp.strip().startswith("=")

    def test_showboard(self, engine):
        resp = engine.handle_command("showboard")
        assert resp.strip().startswith("=")

    def test_undo(self, engine):
        engine.handle_command("play B E5")
        resp = engine.handle_command("undo")
        assert resp.strip() == "="
        assert engine.board.grid[4, 4] == Color.EMPTY


# ===========================================================================
# MoveOptimizer
# ===========================================================================

class TestMoveOptimizer:
    def test_optimize_returns_legal_move(self, empty_board_9, optimizer_no_model):
        move, meta = optimizer_no_model.optimize_move(empty_board_9)
        assert "gtp_coord" in meta
        assert "winrate" in meta
        if move != PASS_MOVE:
            r, c = move
            assert 0 <= r < 9 and 0 <= c < 9

    def test_atari_capture_prioritized(self, atari_board, optimizer_no_model):
        captures = optimizer_no_model.find_immediate_atari_captures(atari_board, Color.BLACK)
        assert (2, 3) in captures

    def test_endangered_save_detected(self):
        b = Board(size=9)
        b.grid[4, 4] = Color.BLACK
        b.grid[3, 4] = Color.WHITE
        b.grid[5, 4] = Color.WHITE
        b.grid[4, 3] = Color.WHITE
        mcts = MCTS(num_simulations=10)
        opt = MoveOptimizer(mcts)
        saves = opt.find_endangered_group_saves(b, Color.BLACK)
        assert (4, 5) in saves

    @pytest.mark.parametrize("r,c,expected", [
        (0, 0, True),   # Solid black corner eye
        (0, 0, False),  # Broken by opponent diagonal
    ])
    def test_true_eye_detection(self, r, c, expected):
        b = Board(size=9)
        b.grid[0, 1] = Color.BLACK
        b.grid[1, 0] = Color.BLACK
        if expected:
            b.grid[1, 1] = Color.BLACK
        else:
            b.grid[1, 1] = Color.WHITE
        mcts = MCTS(num_simulations=10)
        opt = MoveOptimizer(mcts)
        assert opt.is_true_eye(b, r, c, Color.BLACK) is expected


# ===========================================================================
# LadderSolver
# ===========================================================================

class TestLadderSolver:
    def test_simple_ladder(self):
        b = Board(size=9)
        b.grid[1, 1] = Color.WHITE
        b.grid[0, 1] = Color.BLACK
        b.grid[1, 0] = Color.BLACK
        b.grid[0, 2] = Color.BLACK
        b.grid[2, 0] = Color.BLACK
        solver = LadderSolver()
        can_capture, seq = solver.is_ladder_capturable(b, (1, 1), Color.BLACK)
        assert can_capture is True
        assert len(seq) > 0

    def test_non_ladder(self):
        b = Board(size=9)
        b.grid[4, 4] = Color.WHITE  # Center stone with 4 liberties
        solver = LadderSolver()
        can_capture, seq = solver.is_ladder_capturable(b, (4, 4), Color.BLACK)
        assert can_capture is False


# ===========================================================================
# JosekiBook
# ===========================================================================

class TestJosekiBook:
    def test_9x9_opening(self):
        book = JosekiBook()
        b = Board(size=9)
        mv = book.get_book_move(b)
        valid_9 = {
            book._transform_coord(seq[0], 9, k, flip)
            for seq in book.openings_9x9
            for k in range(4)
            for flip in [False, True]
        }
        assert mv in valid_9
        assert b.is_legal(mv, Color.BLACK)

    def test_19x19_opening(self):
        book = JosekiBook()
        b = Board(size=19)
        mv = book.get_book_move(b)
        valid_19 = {
            book._transform_coord(seq[0], 19, k, flip)
            for seq in book.openings_19x19
            for k in range(4)
            for flip in [False, True]
        }
        assert mv in valid_19
        assert b.is_legal(mv, Color.BLACK)

    def test_mid_game_returns_none_or_move(self):
        book = JosekiBook()
        b = Board(size=9)
        # Play 10 moves to get past book
        for i in range(10):
            b.play((i % 9, (i * 3) % 9), Color.BLACK if i % 2 == 0 else Color.WHITE)
        mv = book.get_book_move(b)
        # Either a move or None (out of book)
        assert mv is None or (isinstance(mv, tuple) and len(mv) == 2)
