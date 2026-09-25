"""Deep Reinforcement Learning Trainer — AlphaZero-style Self-Play + Training + Gating."""

from __future__ import annotations
import os
import copy
import math
import random
import time
import logging
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gobot_engine.board import Board, Color, Move, PASS_MOVE
from gobot_engine.neural_net import GoResNet, GoTransformerNet, DualHeadLoss
from gobot_engine.mcts import MCTS
from .dataset import GoDataset, GoDataPoint

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Circular replay buffer for storing self-play training data.

    Stores GoDataPoint entries with configurable maximum size.
    Oldest entries are dropped when capacity is exceeded (FIFO).
    Supports random sampling for training batch construction.
    """

    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer: deque[GoDataPoint] = deque(maxlen=max_size)

    def add_game(self, game_data: List[GoDataPoint]) -> None:
        """Adds a full game's worth of data points to the buffer."""
        for dp in game_data:
            self.buffer.append(dp)

    def sample(self, batch_size: int) -> List[GoDataPoint]:
        """Samples a random batch of data points from the buffer."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def to_dataset(self, augment: bool = True) -> GoDataset:
        """Converts the entire buffer into a GoDataset for DataLoader usage."""
        return GoDataset(list(self.buffer), augment_symmetry=augment)

    def __len__(self) -> int:
        return len(self.buffer)


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights that is updated as:
        shadow = decay * shadow + (1 - decay) * current_params
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Updates shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Copies shadow weights into the model's parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])


class RLTrainer:
    """Deep Reinforcement Learning Trainer inspired by AlphaZero.

    Implements the full RL loop:
    1. Self-play: generate games with MCTS guided by current best network
    2. Training: update network on self-play data with policy + value loss
    3. Evaluation: pit new model vs best model in head-to-head matches
    4. Gating: accept new model only if it wins > threshold
    """

    def __init__(
        self,
        model: Union[GoResNet, GoTransformerNet],
        board_size: int = 9,
        num_simulations: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ema_decay: float = 0.999,
        replay_buffer_size: int = 50000,
        device: str = "cpu",
    ):
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.device = torch.device(
            device if torch.cuda.is_available() and "cuda" in device else "cpu"
        )

        # Current model being trained
        self.current_model = model.to(self.device)

        # Best model (used for self-play data generation)
        self.best_model = copy.deepcopy(model).to(self.device)
        self.best_model.eval()

        # Optimizer with L2 regularization (weight_decay)
        self.optimizer = optim.AdamW(
            self.current_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = DualHeadLoss(value_weight=1.0)
        self.ema = EMA(self.current_model, decay=ema_decay)
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

        # Tracking
        self.elo_estimate: float = 1000.0
        self.iteration_history: List[Dict[str, Any]] = []

    def _self_play_phase(
        self,
        num_games: int = 20,
        temp_threshold_moves: int = 15,
    ) -> int:
        """Generates self-play games using MCTS guided by the best model.

        Returns the total number of new training positions generated.
        """
        self.best_model.eval()

        mcts = MCTS(
            model=self.best_model,
            num_simulations=self.num_simulations,
            dirichlet_alpha=0.03 if self.board_size <= 9 else 0.01,
            dirichlet_epsilon=0.25,
            device=str(self.device),
        )

        total_positions = 0

        for g in range(num_games):
            board = Board(size=self.board_size, komi=7.5)
            game_history: List[Tuple[np.ndarray, np.ndarray, int]] = []
            move_count = 0
            max_moves = self.board_size * self.board_size * 3

            while not board.is_game_over and move_count < max_moves:
                color = board.to_move
                temp = 1.0 if move_count < temp_threshold_moves else 0.2
                feat = board.to_feature_tensor(color).numpy()

                move, action_probs, _ = mcts.search(
                    board, temperature=temp, add_noise=True
                )

                game_history.append((feat, action_probs, color))
                board.play(move, color)
                move_count += 1

            # Determine winner and assign values
            score = board.calculate_area_score()
            winner = score["winner"]

            game_data: List[GoDataPoint] = []
            for feat, pi, color in game_history:
                if winner == color:
                    z = 1.0
                elif winner == Color.opponent(color):
                    z = -1.0
                else:
                    z = 0.0

                game_data.append(
                    GoDataPoint(
                        feature_tensor=feat,
                        target_action=pi,  # MCTS visit distribution
                        target_value=z,
                        board_size=self.board_size,
                    )
                )

            self.replay_buffer.add_game(game_data)
            total_positions += len(game_data)

            if (g + 1) % max(1, num_games // 5) == 0:
                print(f"  Self-play: {g + 1}/{num_games} games | "
                      f"Buffer: {len(self.replay_buffer)} positions")

        return total_positions

    def _training_phase(
        self,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Trains the current model on replay buffer data.

        Uses KL-divergence for policy (via soft cross-entropy) and MSE for value.
        Returns training metrics.
        """
        if len(self.replay_buffer) < batch_size:
            print("  Warning: Not enough data in replay buffer for training.")
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        self.current_model.train()
        dataset = self.replay_buffer.to_dataset(augment=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            for x, target_policy, target_value in dataloader:
                x = x.to(self.device)
                target_policy = target_policy.to(self.device)
                target_value = target_value.to(self.device)

                self.optimizer.zero_grad()
                pred_policy_logits, pred_value = self.current_model(x)

                loss, p_loss, v_loss = self.criterion(
                    pred_policy_logits, pred_value, target_policy, target_value
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.current_model.parameters(), max_norm=1.0
                )
                self.optimizer.step()
                self.ema.update(self.current_model)

                bs = x.size(0)
                total_samples += bs
                epoch_samples += bs
                total_loss += loss.item() * bs
                total_p_loss += p_loss.item() * bs
                total_v_loss += v_loss.item() * bs
                epoch_loss += loss.item() * bs

            avg_epoch_loss = epoch_loss / max(1, epoch_samples)
            print(f"  Train Epoch {epoch + 1}/{epochs}: Loss = {avg_epoch_loss:.4f}")

        return {
            "loss": total_loss / max(1, total_samples),
            "policy_loss": total_p_loss / max(1, total_samples),
            "value_loss": total_v_loss / max(1, total_samples),
        }

    def _evaluation_phase(
        self,
        eval_games: int = 10,
        max_moves_per_game: int = 200,
    ) -> float:
        """Pits the new (EMA) model against the current best.

        Returns the win rate of the new model.
        """
        # Create evaluation model with EMA weights
        eval_model = copy.deepcopy(self.current_model)
        self.ema.apply_shadow(eval_model)
        eval_model.eval()
        self.best_model.eval()

        # Reduce simulations for faster evaluation
        eval_sims = max(10, self.num_simulations // 4)
        new_mcts = MCTS(model=eval_model, num_simulations=eval_sims, device=str(self.device))
        best_mcts = MCTS(model=self.best_model, num_simulations=eval_sims, device=str(self.device))

        new_wins = 0
        draws = 0

        for g in range(eval_games):
            board = Board(size=self.board_size, komi=7.5)
            # Alternate colors
            new_color = Color.BLACK if g % 2 == 0 else Color.WHITE
            best_color = Color.opponent(new_color)
            move_count = 0

            while not board.is_game_over and move_count < max_moves_per_game:
                if board.to_move == new_color:
                    move, _, _ = new_mcts.search(board, temperature=0.0)
                else:
                    move, _, _ = best_mcts.search(board, temperature=0.0)

                board.play(move)
                move_count += 1

            score = board.calculate_area_score()
            winner = score["winner"]

            if winner == new_color:
                new_wins += 1
            elif winner != best_color:
                draws += 1

        win_rate = new_wins / max(1, eval_games)
        print(f"  Evaluation: {new_wins}W / {eval_games - new_wins - draws}L / {draws}D "
              f"(win rate: {win_rate * 100:.1f}%)")

        return win_rate

    def run_rl_training(
        self,
        num_iterations: int = 10,
        games_per_iteration: int = 20,
        training_epochs: int = 5,
        batch_size: int = 32,
        eval_games: int = 10,
        win_threshold: float = 0.55,
        checkpoint_dir: str = "checkpoints",
        output_model_name: str = "rl_gobot_model.pt",
    ) -> Dict[str, Any]:
        """Runs the full AlphaZero-style RL training loop.

        Each iteration:
        1. Self-play: generate games with MCTS + best model
        2. Train: update current model on replay buffer
        3. Evaluate: pit new model vs best in head-to-head
        4. Gate: accept new model only if it wins > win_threshold

        Args:
            num_iterations: Number of RL loop iterations
            games_per_iteration: Self-play games per iteration
            training_epochs: Training epochs per iteration
            batch_size: Training batch size
            eval_games: Number of evaluation games for gating
            win_threshold: Win rate required to accept new model
            checkpoint_dir: Directory for saving checkpoints
            output_model_name: Filename for the best model checkpoint

        Returns:
            Summary dict with training history and final metrics
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, output_model_name)

        print(f"\n{'='*60}")
        print(f"  Starting AlphaZero RL Training")
        print(f"  {num_iterations} iterations × {games_per_iteration} games/iter")
        print(f"  Buffer size: {self.replay_buffer.max_size} | "
              f"Gating threshold: {win_threshold*100:.0f}%")
        print(f"{'='*60}\n")

        for iteration in range(1, num_iterations + 1):
            t0 = time.time()
            print(f"\n{'─'*50}")
            print(f"  RL Iteration {iteration}/{num_iterations}")
            print(f"{'─'*50}")

            # 1. Self-play
            print(f"\n  [Phase 1] Self-Play ({games_per_iteration} games)...")
            new_positions = self._self_play_phase(
                num_games=games_per_iteration,
            )
            print(f"  Generated {new_positions} positions (buffer: {len(self.replay_buffer)})")

            # 2. Training
            print(f"\n  [Phase 2] Training ({training_epochs} epochs)...")
            train_metrics = self._training_phase(
                epochs=training_epochs,
                batch_size=batch_size,
            )

            # 3. Evaluation + Gating
            print(f"\n  [Phase 3] Evaluation ({eval_games} games)...")
            win_rate = self._evaluation_phase(eval_games=eval_games)

            accepted = win_rate > win_threshold
            if accepted:
                # Accept new model
                self.ema.apply_shadow(self.current_model)
                self.best_model.load_state_dict(self.current_model.state_dict())
                self.best_model.eval()

                # Update ELO estimate
                if win_rate > 0.5:
                    elo_gain = 400 * math.log10(
                        max(0.01, win_rate) / max(0.01, 1 - win_rate)
                    )
                    self.elo_estimate += max(5, min(50, elo_gain))

                # Save best model checkpoint
                if hasattr(self.best_model, 'save_checkpoint'):
                    self.best_model.save_checkpoint(best_path, extra_meta={
                        "iteration": iteration,
                        "win_rate": win_rate,
                        "elo_estimate": self.elo_estimate,
                        "train_loss": train_metrics["loss"],
                    })
                else:
                    torch.save(self.best_model.state_dict(), best_path)

                print(f"  ✓ Model ACCEPTED (win rate {win_rate*100:.1f}% > {win_threshold*100:.0f}%)")
            else:
                # Reject: revert to best model
                self.current_model.load_state_dict(self.best_model.state_dict())
                self.ema = EMA(self.current_model, decay=self.ema.decay)
                print(f"  ✗ Model REJECTED (win rate {win_rate*100:.1f}% ≤ {win_threshold*100:.0f}%)")

            elapsed = time.time() - t0
            iter_result = {
                "iteration": iteration,
                "accepted": accepted,
                "win_rate": win_rate,
                "elo_estimate": self.elo_estimate,
                "train_loss": train_metrics["loss"],
                "policy_loss": train_metrics["policy_loss"],
                "value_loss": train_metrics["value_loss"],
                "buffer_size": len(self.replay_buffer),
                "new_positions": new_positions,
                "time_sec": round(elapsed, 1),
            }
            self.iteration_history.append(iter_result)

            print(f"\n  ELO: ~{self.elo_estimate:.0f} | Loss: {train_metrics['loss']:.4f} | "
                  f"Time: {elapsed:.1f}s")

        # Final summary
        print(f"\n{'='*60}")
        print(f"  RL Training Complete!")
        print(f"  Final ELO estimate: ~{self.elo_estimate:.0f}")
        print(f"  Best model saved to: {best_path}")
        accepted_count = sum(1 for h in self.iteration_history if h["accepted"])
        print(f"  Models accepted: {accepted_count}/{num_iterations}")
        print(f"{'='*60}\n")

        return {
            "final_elo": self.elo_estimate,
            "checkpoint_path": best_path,
            "history": self.iteration_history,
            "models_accepted": accepted_count,
        }
