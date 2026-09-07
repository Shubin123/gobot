"""Hyperparameter Optimizer and Search Loop for GoBot Engine."""

from __future__ import annotations
import os
import json
import itertools
from typing import Dict, Any, List, Optional

from gobot_engine.neural_net import GoResNet
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from benchmark_suite import BenchmarkSuite


PARAM_GRID = {
    "c_puct": [1.4, 2.0, 2.8],
    "dirichlet_epsilon": [0.15, 0.25],
    "use_joseki_book": [True],
    "enable_tactics": [True],
    "num_simulations": [30, 60],
}


class HyperparameterOptimizer:
    """Explores search parameter space and finds the winning hyperparameter setup."""

    def __init__(self, model_path: Optional[str] = None, board_size: int = 9):
        self.board_size = board_size
        self.model = GoResNet.load_checkpoint(model_path) if model_path and os.path.exists(model_path) else None

    def search_optimal_params(
        self,
        trials: int = 4,
        output_json: str = "checkpoints/optimal_hyperparams.json",
    ) -> Dict[str, Any]:
        """Runs search across param combinations and tracks performance against base metric."""
        keys = list(PARAM_GRID.keys())
        all_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

        # Sample up to `trials` combinations
        if len(all_combos) > trials:
            selected_combos = all_combos[:trials]
        else:
            selected_combos = all_combos

        print("=" * 65)
        print(f"       GoBot Hyperparameter Optimization (Evaluating {len(selected_combos)} Configs)")
        print("=" * 65)

        best_score = -1.0
        best_config = None
        best_report = None
        results = []

        for idx, cfg in enumerate(selected_combos, 1):
            print(f"\n--- [Trial {idx}/{len(selected_combos)}] Testing Config: c_puct={cfg['c_puct']}, eps={cfg['dirichlet_epsilon']}, sims={cfg['num_simulations']} ---")

            mcts = MCTS(
                model=self.model,
                c_puct=cfg["c_puct"],
                dirichlet_epsilon=cfg["dirichlet_epsilon"],
                num_simulations=cfg["num_simulations"],
            )
            optimizer = MoveOptimizer(
                mcts=mcts,
                enable_tactics=cfg["enable_tactics"],
                use_joseki_book=cfg["use_joseki_book"],
            )

            suite = BenchmarkSuite(optimizer, board_size=self.board_size)
            report = suite.run_full_benchmark()
            score = report["overall_performance_index"]
            winrate = report["vs_greedy_winrate_pct"]

            print(f" Result: OPI = {score}/100.0 | Vs Greedy Winrate = {winrate}% | Tsumego = {report['tsumego_accuracy_pct']}%")

            results.append({"config": cfg, "report": report})

            if score > best_score:
                best_score = score
                best_config = cfg
                best_report = report

        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        final_summary = {
            "best_hyperparameters": best_config,
            "best_performance_index": best_score,
            "best_report": best_report,
            "all_trials": results,
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)

        print("\n" + "=" * 65)
        print(" [+] Optimal Hyperparameter Configuration Discovered:")
        print(json.dumps(best_config, indent=2))
        print(f" Best OPI Score: {best_score} / 100.0")
        print(f" Saved to: {output_json}")
        print("=" * 65 + "\n")

        return final_summary


if __name__ == "__main__":
    chk = "checkpoints/winning_gobot_model.pt"
    if not os.path.exists(chk):
        chk = "checkpoints/gobot_model.pt" if os.path.exists("checkpoints/gobot_model.pt") else None
    opt = HyperparameterOptimizer(model_path=chk, board_size=9)
    opt.search_optimal_params(trials=3)
