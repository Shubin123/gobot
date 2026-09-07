"""Modulated High-Performance Auto-Trainer for Winning Go Bots."""

from __future__ import annotations
import os
import time
import math
from typing import Dict, Any, Optional, List
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from gobot_engine.board import Board
from gobot_engine.neural_net import GoResNet, DualHeadLoss
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from .dataset import GoDataset
from .curated_data import build_curated_master_dataset
from benchmark_suite import BenchmarkSuite, print_benchmark_report


class AutoTrainer:
    """Trains and optimizes Go ResNet models using curated master datasets & cosine scheduling."""

    def __init__(
        self,
        board_size: int = 9,
        num_blocks: int = 4,
        num_filters: int = 48,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.board_size = board_size
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.model = GoResNet(
            board_size=board_size,
            in_channels=8,
            num_filters=num_filters,
            num_blocks=num_blocks,
        ).to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = DualHeadLoss(value_weight=1.2)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def run_modulated_training(
        self,
        epochs: int = 6,
        batch_size: int = 32,
        checkpoint_dir: str = "checkpoints",
        output_model_name: str = "winning_gobot_model.pt",
    ) -> Dict[str, Any]:
        """Runs the optimized training loop on curated master datasets."""
        print(f"--- Ingesting Curated Pro & Tactical Datasets ({self.board_size}x{self.board_size}) ---")
        dataset = build_curated_master_dataset(board_size=self.board_size)
        print(f"Ingested {len(dataset)} augmented pro game and tsumego positions.")

        val_size = max(4, int(len(dataset) * 0.15))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, output_model_name)

        history = []
        best_val_loss = float("inf")

        print(f"\n[>>>] Starting Modulated Training for {epochs} Epochs on {self.device.type.upper()}...")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            train_loss = 0.0
            train_p_loss = 0.0
            train_v_loss = 0.0
            correct = 0
            total = 0

            for x, target_p, target_v in train_loader:
                x = x.to(self.device)
                target_p = target_p.to(self.device)
                target_v = target_v.to(self.device)

                self.optimizer.zero_grad()
                pred_p, pred_v = self.model(x)
                loss, p_loss, v_loss = self.criterion(pred_p, pred_v, target_p, target_v)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()

                bs = x.size(0)
                total += bs
                train_loss += loss.item() * bs
                train_p_loss += p_loss.item() * bs
                train_v_loss += v_loss.item() * bs

                pred_acts = torch.argmax(pred_p, dim=1)
                target_acts = torch.argmax(target_p, dim=1) if target_p.dim() > 1 else target_p
                correct += (pred_acts == target_acts).sum().item()

            scheduler.step()

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x, target_p, target_v in val_loader:
                    x = x.to(self.device)
                    target_p = target_p.to(self.device)
                    target_v = target_v.to(self.device)
                    pred_p, pred_v = self.model(x)
                    loss, _, _ = self.criterion(pred_p, pred_v, target_p, target_v)

                    bs = x.size(0)
                    val_total += bs
                    val_loss += loss.item() * bs
                    pred_acts = torch.argmax(pred_p, dim=1)
                    target_acts = torch.argmax(target_p, dim=1) if target_p.dim() > 1 else target_p
                    val_correct += (pred_acts == target_acts).sum().item()

            epoch_time = time.time() - t0
            t_loss = train_loss / total
            t_acc = (correct / total) * 100
            v_loss = val_loss / val_total
            v_acc = (val_correct / val_total) * 100

            print(f" Epoch {epoch:2d}/{epochs} | Train Loss: {t_loss:.4f} (Acc: {t_acc:.1f}%) | Val Loss: {v_loss:.4f} (Acc: {v_acc:.1f}%) | Time: {epoch_time:.2f}s")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                self.model.save_checkpoint(checkpoint_path, extra_meta={"val_loss": v_loss, "val_acc": v_acc})

        print(f"\n[OK] Winning Model Checkpoint saved to: {checkpoint_path}")

        # Evaluate on Benchmark Suite
        print(f"\n[>>>] Running Base Metric Benchmark on the Trained Model...")
        mcts_engine = MCTS(model=self.model, num_simulations=40)
        optimizer = MoveOptimizer(mcts_engine)
        suite = BenchmarkSuite(optimizer, board_size=self.board_size)
        report = suite.run_full_benchmark()
        print_benchmark_report(report)

        return {
            "checkpoint_path": checkpoint_path,
            "best_val_loss": best_val_loss,
            "benchmark_report": report,
        }
