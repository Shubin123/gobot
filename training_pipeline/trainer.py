"""Model Trainer and Fine-Tuning Loop for Go ResNet."""

from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from gobot_engine.board import Board, Color, Move, PASS_MOVE
from gobot_engine.neural_net import GoResNet, DualHeadLoss
from gobot_engine.mcts import MCTS
from .dataset import GoDataset


class GoTrainer:
    """Trains and fine-tunes GoResNet models with training metrics and baseline evaluation."""

    def __init__(
        self,
        model: GoResNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        value_weight: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.model.to(self.device)
        self.criterion = DualHeadLoss(value_weight=value_weight)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        correct_top1 = 0
        total_samples = 0

        for x, target_policy, target_value in dataloader:
            x = x.to(self.device)
            target_policy = target_policy.to(self.device)
            target_value = target_value.to(self.device)

            self.optimizer.zero_grad()
            pred_policy_logits, pred_value = self.model(x)

            loss, p_loss, v_loss = self.criterion(
                pred_policy_logits, pred_value, target_policy, target_value
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            batch_size = x.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_p_loss += p_loss.item() * batch_size
            total_v_loss += v_loss.item() * batch_size

            # Calculate top-1 accuracy
            if target_policy.dim() > 1:
                target_actions = torch.argmax(target_policy, dim=1)
            else:
                target_actions = target_policy
            pred_actions = torch.argmax(pred_policy_logits, dim=1)
            correct_top1 += (pred_actions == target_actions).sum().item()

        return {
            "loss": total_loss / max(1, total_samples),
            "policy_loss": total_p_loss / max(1, total_samples),
            "value_loss": total_v_loss / max(1, total_samples),
            "top1_accuracy": correct_top1 / max(1, total_samples),
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        correct_top1 = 0
        total_samples = 0

        with torch.no_grad():
            for x, target_policy, target_value in dataloader:
                x = x.to(self.device)
                target_policy = target_policy.to(self.device)
                target_value = target_value.to(self.device)

                pred_policy_logits, pred_value = self.model(x)
                loss, p_loss, v_loss = self.criterion(
                    pred_policy_logits, pred_value, target_policy, target_value
                )

                batch_size = x.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size
                total_p_loss += p_loss.item() * batch_size
                total_v_loss += v_loss.item() * batch_size

                if target_policy.dim() > 1:
                    target_actions = torch.argmax(target_policy, dim=1)
                else:
                    target_actions = target_policy
                pred_actions = torch.argmax(pred_policy_logits, dim=1)
                correct_top1 += (pred_actions == target_actions).sum().item()

        return {
            "val_loss": total_loss / max(1, total_samples),
            "val_policy_loss": total_p_loss / max(1, total_samples),
            "val_value_loss": total_v_loss / max(1, total_samples),
            "val_top1_accuracy": correct_top1 / max(1, total_samples),
        }

    def fit(
        self,
        dataset: GoDataset,
        epochs: int = 5,
        batch_size: int = 32,
        val_split: float = 0.1,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "gobot_model.pt",
    ) -> List[Dict[str, float]]:
        """Fits model on dataset with validation split and checkpoint saving."""
        if len(dataset) == 0:
            raise ValueError("Cannot train on an empty dataset.")

        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        if val_size > 0:
            train_set, val_set = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            val_loader = None

        history: List[Dict[str, float]] = []
        best_val_loss = float("inf")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, model_name)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.scheduler.step(val_metrics["val_loss"])
                metrics = {**train_metrics, **val_metrics}
            else:
                metrics = train_metrics

            elapsed = time.time() - t0
            metrics["epoch"] = epoch
            metrics["time_sec"] = elapsed
            history.append(metrics)

            # Save best checkpoint
            current_loss = metrics.get("val_loss", metrics["loss"])
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                self.model.save_checkpoint(save_path, extra_meta=metrics)

        return history

    @staticmethod
    def play_match(
        bot_black: MCTS,
        bot_white: MCTS,
        board_size: int = 19,
        komi: float = 7.5,
        max_moves: int = 300,
    ) -> Dict[str, Any]:
        """Plays a game between two MCTS engines and returns the game result."""
        board = Board(size=board_size, komi=komi)
        move_count = 0

        while not board.is_game_over and move_count < max_moves:
            current_bot = bot_black if board.to_move == Color.BLACK else bot_white
            move, _, _ = current_bot.search(board, temperature=0.0)
            board.play(move)
            move_count += 1

        score = board.calculate_area_score()
        return {
            "winner": score["winner"],
            "result_str": score["result_str"],
            "moves_played": move_count,
            "black_score": score["black_score"],
            "white_score": score["white_score"],
        }
