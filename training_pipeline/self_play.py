"""Self-Play Dataset Generator using Monte Carlo Tree Search."""

from __future__ import annotations
import os
import time
from typing import List, Tuple, Optional
import numpy as np

from gobot_engine.board import Board, Color, Move, PASS_MOVE
from gobot_engine.mcts import MCTS
from gobot_engine.neural_net import GoResNet
from .dataset import GoDataset, GoDataPoint


class SelfPlayWorker:
    """Generates self-play games for reinforcement learning & curriculum training."""

    def __init__(
        self,
        model: Optional[GoResNet] = None,
        board_size: int = 19,
        komi: float = 7.5,
        num_simulations: int = 50,
        temp_threshold_moves: int = 15,
        c_puct: float = 2.0,
        device: str = "cpu",
    ):
        self.board_size = board_size
        self.komi = komi
        self.num_simulations = num_simulations
        self.temp_threshold_moves = temp_threshold_moves
        self.mcts = MCTS(
            model=model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
        )

    def play_single_game(
        self,
        max_moves: int = 300,
        save_sgf_path: Optional[str] = None,
    ) -> List[GoDataPoint]:
        """Plays one complete self-play game and returns recorded training data points."""
        board = Board(size=self.board_size, komi=self.komi)
        game_history: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (feat, pi, color)
        move_count = 0

        while not board.is_game_over and move_count < max_moves:
            color = board.to_move
            temp = 1.0 if move_count < self.temp_threshold_moves else 0.2
            feat = board.to_feature_tensor(color).numpy()

            move, action_probs, _ = self.mcts.search(board, temperature=temp, add_noise=True)

            game_history.append((feat, action_probs, color))
            board.play(move, color)
            move_count += 1

        # Determine winner
        score = board.calculate_area_score()
        winner = score["winner"]

        # Build training points with true terminal value z
        datapoints: List[GoDataPoint] = []
        for feat, pi, color in game_history:
            if winner == color:
                z = 1.0
            elif winner == Color.opponent(color):
                z = -1.0
            else:
                z = 0.0

            datapoints.append(
                GoDataPoint(
                    feature_tensor=feat,
                    target_action=pi,
                    target_value=z,
                    board_size=self.board_size,
                )
            )

        if save_sgf_path:
            self._save_game_to_sgf(board, score["result_str"], save_sgf_path)

        return datapoints

    def _save_game_to_sgf(self, board: Board, result_str: str, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        sgf_moves = []
        for color, move in board.move_history:
            col_ch = "B" if color == Color.BLACK else "W"
            sgf_moves.append(f";{col_ch}[{Move.to_sgf(move)}]")

        sgf_content = (
            f"(;GM[1]FF[4]SZ[{self.board_size}]KM[{self.komi:.1f}]"
            f"PB[GoBot_SelfPlay]PW[GoBot_SelfPlay]RE[{result_str}]\n"
            + "".join(sgf_moves)
            + ")\n"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(sgf_content)

    def generate_batch(
        self,
        num_games: int = 5,
        max_moves_per_game: int = 200,
        sgf_output_dir: Optional[str] = None,
    ) -> GoDataset:
        """Generates multiple self-play games and compiles them into a GoDataset."""
        dataset = GoDataset()
        for g in range(num_games):
            sgf_path = (
                os.path.join(sgf_output_dir, f"selfplay_game_{int(time.time())}_{g}.sgf")
                if sgf_output_dir
                else None
            )
            dps = self.play_single_game(max_moves=max_moves_per_game, save_sgf_path=sgf_path)
            dataset.extend(dps)
        return dataset
