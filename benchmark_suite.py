"""Comprehensive Base Metric & Benchmark Suite for Go Bot Engines."""

from __future__ import annotations
import os
import sys
import math
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Clean SSL keylog file in case of Windows permissions
os.environ.pop("SSLKEYLOGFILE", None)

from gobot_engine.board import Board, Color, Move, PASS_MOVE
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from gobot_engine.neural_net import GoResNet
from gobot_engine.ladder import LadderSolver
from training_pipeline.curated_data import TSUMEGO_PROBLEMS_9x9


class BaselineEngine:
    """Configurable baseline engine for benchmarking."""

    def __init__(self, mode: str = "random", num_simulations: int = 40):
        self.mode = mode
        self.mcts = MCTS(num_simulations=num_simulations) if "mcts" in mode else None
        self.optimizer = MoveOptimizer(self.mcts) if self.mcts else None

    def genmove(self, board: Board, color: int) -> Tuple[int, int]:
        legal_moves = board.get_legal_moves(color)
        if not legal_moves:
            return PASS_MOVE

        if self.mode == "random":
            # Filter pass if board not empty
            non_pass = [m for m in legal_moves if not Move.is_pass(m)]
            return non_pass[np.random.randint(len(non_pass))] if non_pass else PASS_MOVE

        elif self.mode == "greedy_tactical":
            # Check immediate captures
            temp_opt = MoveOptimizer(MCTS(num_simulations=5))
            caps = temp_opt.find_immediate_atari_captures(board, color)
            if caps:
                return caps[0]
            # Avoid self-atari
            safe_moves = [m for m in legal_moves if not temp_opt.is_self_atari_blunder(board, m, color)]
            return safe_moves[0] if safe_moves else legal_moves[0]

        elif "mcts" in self.mode and self.optimizer:
            mv, _ = self.optimizer.optimize_move(board)
            return mv

        return legal_moves[0]


class BenchmarkSuite:
    """Evaluates engine strength across tactical, opening, and tournament matches."""

    def __init__(self, engine_optimizer: MoveOptimizer, board_size: int = 9):
        self.optimizer = engine_optimizer
        self.board_size = board_size
        self.ladder_solver = LadderSolver(max_depth=50)

    def run_tsumego_benchmark(self) -> Dict[str, Any]:
        """Tests tactical life & death solving rate."""
        total = len(TSUMEGO_PROBLEMS_9x9)
        solved = 0
        details = []

        for prob in TSUMEGO_PROBLEMS_9x9:
            board = Board(size=self.board_size)
            for col, coord in prob["stones"]:
                if board.in_bounds(coord[0], coord[1]):
                    board.grid[coord[0], coord[1]] = col
            board.to_move = prob["to_move"]

            move, meta = self.optimizer.optimize_move(board)
            expected = prob["vital_move"]
            is_correct = (move == expected)
            if is_correct:
                solved += 1

            details.append({
                "problem": prob["name"],
                "expected": Move.to_gtp(expected, self.board_size),
                "predicted": meta["gtp_coord"],
                "is_correct": is_correct,
            })

        return {
            "total_problems": total,
            "solved": solved,
            "accuracy_pct": round((solved / max(1, total)) * 100, 1),
            "details": details,
        }

    def run_ladder_benchmark(self) -> Dict[str, Any]:
        """Tests ladder tactical recognition."""
        board = Board(size=self.board_size)
        board.grid[1, 1] = Color.WHITE
        board.grid[0, 1] = Color.BLACK
        board.grid[1, 0] = Color.BLACK
        board.grid[0, 2] = Color.BLACK
        board.grid[2, 0] = Color.BLACK
        can_capture, seq = self.ladder_solver.is_ladder_capturable(board, (1, 1), Color.BLACK)

        return {
            "ladder_test_passed": can_capture,
            "simulated_depth": len(seq),
            "forcing_sequence": [Move.to_gtp(m, 9) for m in seq],
        }

    def run_tournament_matches(
        self,
        opponent_mode: str = "greedy_tactical",
        num_games: int = 6,
        max_moves_per_game: int = 120,
    ) -> Dict[str, Any]:
        """Runs head-to-head matches against a baseline engine."""
        opponent = BaselineEngine(mode=opponent_mode)
        bot_wins = 0
        opponent_wins = 0
        draws = 0
        total_margin = 0.0

        for g in range(num_games):
            board = Board(size=self.board_size, komi=7.5)
            # Alternate colors: Bot is Black on even games, White on odd games
            bot_color = Color.BLACK if g % 2 == 0 else Color.WHITE
            opp_color = Color.opponent(bot_color)

            moves_count = 0
            while not board.is_game_over and moves_count < max_moves_per_game:
                if board.to_move == bot_color:
                    mv, _ = self.optimizer.optimize_move(board)
                else:
                    mv = opponent.genmove(board, opp_color)

                board.play(mv)
                moves_count += 1

            score = board.calculate_area_score()
            winner = score["winner"]
            margin = score["margin"]

            if winner == bot_color:
                bot_wins += 1
                total_margin += abs(margin)
            elif winner == opp_color:
                opponent_wins += 1
                total_margin -= abs(margin)
            else:
                draws += 1

        winrate = round((bot_wins / max(1, num_games)) * 100, 1)

        # Approximate ELO calculation relative to baseline (1000 base)
        if winrate == 100.0:
            est_elo = 1400
        elif winrate == 0.0:
            est_elo = 600
        else:
            w_ratio = (bot_wins + 0.5 * draws) / num_games
            est_elo = round(1000 + 400 * math.log10(max(0.01, w_ratio / max(0.01, 1 - w_ratio))))

        return {
            "opponent": opponent_mode,
            "games_played": num_games,
            "bot_wins": bot_wins,
            "opponent_wins": opponent_wins,
            "draws": draws,
            "winrate_pct": winrate,
            "avg_margin": round(total_margin / max(1, num_games), 1),
            "estimated_elo": est_elo,
        }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Runs the complete suite and compiles metrics."""
        t0 = time.time()
        tsumego_res = self.run_tsumego_benchmark()
        ladder_res = self.run_ladder_benchmark()
        tourn_greedy = self.run_tournament_matches(opponent_mode="greedy_tactical", num_games=6)
        tourn_mcts = self.run_tournament_matches(opponent_mode="mcts_shallow", num_games=4)
        elapsed = time.time() - t0

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "board_size": self.board_size,
            "benchmark_time_sec": round(elapsed, 2),
            "tsumego_accuracy_pct": tsumego_res["accuracy_pct"],
            "ladder_test_passed": ladder_res["ladder_test_passed"],
            "vs_greedy_winrate_pct": tourn_greedy["winrate_pct"],
            "vs_greedy_elo": tourn_greedy["estimated_elo"],
            "vs_mcts_winrate_pct": tourn_mcts["winrate_pct"],
            "vs_mcts_elo": tourn_mcts["estimated_elo"],
            "overall_performance_index": round(
                (tsumego_res["accuracy_pct"] * 0.3)
                + (tourn_greedy["winrate_pct"] * 0.4)
                + (tourn_mcts["winrate_pct"] * 0.3),
                1,
            ),
        }


def print_benchmark_report(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print("        [+] GoBot Engine Base Metric & Performance Report")
    print("=" * 65)
    print(f" Board Size: {metrics['board_size']}x{metrics['board_size']} | Completed In: {metrics['benchmark_time_sec']}s")
    print("-" * 65)
    print(f"  * Tsumego (Life & Death) Accuracy: {metrics['tsumego_accuracy_pct']}%")
    print(f"  * Ladder Tactical Solver:          {'[PASSED]' if metrics['ladder_test_passed'] else '[FAILED]'}")
    print(f"  * Vs Greedy-Tactical Bot Winrate:  {metrics['vs_greedy_winrate_pct']}% (Est. ELO: {metrics['vs_greedy_elo']})")
    print(f"  * Vs Standard MCTS Bot Winrate:   {metrics['vs_mcts_winrate_pct']}% (Est. ELO: {metrics['vs_mcts_elo']})")
    print("-" * 65)
    print(f"  * Overall Performance Index (OPI):  {metrics['overall_performance_index']} / 100.0")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    # Test benchmark run
    default_mcts = MCTS(num_simulations=40)
    optimizer = MoveOptimizer(default_mcts)
    suite = BenchmarkSuite(optimizer, board_size=9)
    report = suite.run_full_benchmark()
    print_benchmark_report(report)
