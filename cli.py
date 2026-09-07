"""Command Line Interface for GoBot Engine, Training, GTP, and Server."""

import os
import sys
import argparse
import uvicorn

from gobot_engine.board import Board, Color, Move, PASS_MOVE, RESIGN_MOVE
from gobot_engine.neural_net import GoResNet
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from gobot_engine.gtp import GTPEngine
from training_pipeline.curriculum import TrainingCurriculum, CurriculumConfig
from training_pipeline.trainer import GoTrainer
from training_pipeline.self_play import SelfPlayWorker
from online_extension.server import create_app


def run_gtp(args):
    model = GoResNet.load_checkpoint(args.model) if args.model and os.path.exists(args.model) else None
    engine = GTPEngine(
        board_size=args.board_size,
        komi=args.komi,
        model=model,
        num_simulations=args.simulations,
    )
    print(f"GoBot GTP Engine initialized (size={args.board_size}, komi={args.komi}). Awaiting commands...", file=sys.stderr)
    engine.run_stdio_loop()


def run_server(args):
    app = create_app(model_path=args.model)
    print(f"Starting GoBot Extension API & Dashboard at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def run_train(args):
    print(f"--- GoBot Curriculum Training Pipeline ---")
    config = CurriculumConfig(
        sources=args.sources,
        board_size=args.board_size,
        min_rank=args.min_rank,
        game_phase=args.game_phase,
        winner_perspective_only=args.winner_only,
        max_games=args.max_games,
    )

    curriculum = TrainingCurriculum(config)
    dataset, stats = curriculum.build_dataset()

    print(f"Found {stats['files_found']} SGF files.")
    print(f"Parsed {stats['total_games_parsed']} games -> Accepted {stats['accepted_games']} games.")
    print(f"Extracted {stats['total_training_positions']} training board positions.")

    if len(dataset) == 0:
        print("Error: No training positions matched curriculum criteria.", file=sys.stderr)
        return

    model = (
        GoResNet.load_checkpoint(args.init_model)
        if args.init_model and os.path.exists(args.init_model)
        else GoResNet(board_size=args.board_size)
    )

    trainer = GoTrainer(model, lr=args.lr)
    history = trainer.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.output_model,
    )

    print(f"\nTraining complete. Model checkpoint saved to {os.path.join(args.checkpoint_dir, args.output_model)}")
    for h in history:
        print(f"Epoch {h['epoch']}: Loss={h['loss']:.4f} | Top1 Acc={h['top1_accuracy']*100:.1f}% | Time={h['time_sec']:.2f}s")


def run_selfplay(args):
    print(f"--- GoBot MCTS Self-Play Generator ---")
    model = GoResNet.load_checkpoint(args.model) if args.model and os.path.exists(args.model) else None
    worker = SelfPlayWorker(
        model=model,
        board_size=args.board_size,
        komi=args.komi,
        num_simulations=args.simulations,
    )
    dataset = worker.generate_batch(
        num_games=args.num_games,
        sgf_output_dir=args.output_dir,
    )
    print(f"Generated {len(dataset)} training positions across {args.num_games} self-play games.")


def run_interactive(args):
    board = Board(size=args.board_size, komi=args.komi)
    model = GoResNet.load_checkpoint(args.model) if args.model and os.path.exists(args.model) else None
    mcts = MCTS(model=model, num_simulations=args.simulations)
    optimizer = MoveOptimizer(mcts)

    human_color = Color.BLACK if args.color.upper() == "B" else Color.WHITE
    bot_color = Color.opponent(human_color)

    print("\n" + "=" * 50)
    print(f" GoBot Interactive Terminal Game ({args.board_size}x{args.board_size})")
    print(f" You: {Color.to_char(human_color)} | Bot: {Color.to_char(bot_color)}")
    print("=" * 50)

    while not board.is_game_over:
        print("\n" + board.render_ascii())
        if board.to_move == human_color:
            try:
                user_in = input(f"Enter move ({Color.to_char(human_color)}) [e.g. D4, pass, resign]: ").strip()
                move = Move.from_gtp(user_in, board.size)
                if not board.play(move, human_color):
                    print("Illegal move! Try again.")
                    continue
            except (ValueError, KeyboardInterrupt) as e:
                print(f"Input error: {e}")
                break
        else:
            print(f"Bot ({Color.to_char(bot_color)}) is thinking...")
            move, meta = optimizer.optimize_move(board)
            board.play(move, bot_color)
            print(f"Bot played: {meta['gtp_coord']} (Winrate: {meta['winrate']*100:.1f}%)")

    score = board.calculate_area_score()
    print("\n" + board.render_ascii())
    print(f"Game Over! Result: {score['result_str']}")


def run_train_pro(args):
    from training_pipeline.auto_trainer import AutoTrainer
    trainer = AutoTrainer(
        board_size=args.board_size,
        num_blocks=args.blocks,
        num_filters=args.filters,
        lr=args.lr,
    )
    trainer.run_modulated_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        output_model_name=args.output_model,
    )


def run_benchmark(args):
    from benchmark_suite import BenchmarkSuite, print_benchmark_report
    model = GoResNet.load_checkpoint(args.model) if args.model and os.path.exists(args.model) else None
    mcts = MCTS(model=model, num_simulations=args.simulations)
    optimizer = MoveOptimizer(mcts)
    suite = BenchmarkSuite(optimizer, board_size=args.board_size)
    report = suite.run_full_benchmark()
    print_benchmark_report(report)


def run_tune(args):
    from hyperopt import HyperparameterOptimizer
    opt = HyperparameterOptimizer(model_path=args.model, board_size=args.board_size)
    opt.search_optimal_params(trials=args.trials, output_json=args.output_json)


def main():
    parser = argparse.ArgumentParser(description="GoBot Engine & Training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # GTP
    p_gtp = subparsers.add_parser("gtp", help="Run GTP 2.0 Engine")
    p_gtp.add_argument("--board-size", type=int, default=19)
    p_gtp.add_argument("--komi", type=float, default=7.5)
    p_gtp.add_argument("--model", type=str, default=None)
    p_gtp.add_argument("--simulations", type=int, default=80)
    p_gtp.set_defaults(func=run_gtp)

    # Server
    p_srv = subparsers.add_parser("server", help="Run FastAPI Online Extension Server")
    p_srv.add_argument("--host", type=str, default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--model", type=str, default=None)
    p_srv.set_defaults(func=run_server)

    # Train (Custom curriculum)
    p_trn = subparsers.add_parser("train", help="Run Curriculum Training on SGFs")
    p_trn.add_argument("--sources", nargs="+", required=True, help="SGF directories or file paths")
    p_trn.add_argument("--board-size", type=int, default=19)
    p_trn.add_argument("--min-rank", type=str, default="any")
    p_trn.add_argument("--game-phase", type=str, default="all")
    p_trn.add_argument("--winner-only", action="store_true")
    p_trn.add_argument("--max-games", type=int, default=None)
    p_trn.add_argument("--epochs", type=int, default=5)
    p_trn.add_argument("--batch-size", type=int, default=32)
    p_trn.add_argument("--lr", type=float, default=1e-3)
    p_trn.add_argument("--init-model", type=str, default=None)
    p_trn.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_trn.add_argument("--output-model", type=str, default="gobot_model.pt")
    p_trn.set_defaults(func=run_train)

    # Train-Pro (Modulated high-performance training on master datasets)
    p_pro = subparsers.add_parser("train-pro", help="Run modulated training on curated master datasets")
    p_pro.add_argument("--board-size", type=int, default=9)
    p_pro.add_argument("--epochs", type=int, default=6)
    p_pro.add_argument("--batch-size", type=int, default=32)
    p_pro.add_argument("--lr", type=float, default=2e-3)
    p_pro.add_argument("--blocks", type=int, default=4)
    p_pro.add_argument("--filters", type=int, default=48)
    p_pro.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_pro.add_argument("--output-model", type=str, default="winning_gobot_model.pt")
    p_pro.set_defaults(func=run_train_pro)

    # Benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run Base Metric Benchmark Suite")
    p_bm.add_argument("--board-size", type=int, default=9)
    p_bm.add_argument("--model", type=str, default="checkpoints/winning_gobot_model.pt")
    p_bm.add_argument("--simulations", type=int, default=40)
    p_bm.set_defaults(func=run_benchmark)

    # Tune (Hyperparameter Optimizer)
    p_tune = subparsers.add_parser("tune", help="Tune MCTS & search hyperparameters")
    p_tune.add_argument("--board-size", type=int, default=9)
    p_tune.add_argument("--model", type=str, default="checkpoints/winning_gobot_model.pt")
    p_tune.add_argument("--trials", type=int, default=4)
    p_tune.add_argument("--output-json", type=str, default="checkpoints/optimal_hyperparams.json")
    p_tune.set_defaults(func=run_tune)

    # Selfplay
    p_slf = subparsers.add_parser("selfplay", help="Run MCTS Self-Play Generator")
    p_slf.add_argument("--num-games", type=int, default=5)
    p_slf.add_argument("--board-size", type=int, default=9)
    p_slf.add_argument("--komi", type=float, default=7.5)
    p_slf.add_argument("--simulations", type=int, default=40)
    p_slf.add_argument("--model", type=str, default=None)
    p_slf.add_argument("--output-dir", type=str, default="selfplay_sgfs")
    p_slf.set_defaults(func=run_selfplay)

    # Interactive
    p_ply = subparsers.add_parser("play", help="Play interactively in terminal")
    p_ply.add_argument("--board-size", type=int, default=9)
    p_ply.add_argument("--komi", type=float, default=7.5)
    p_ply.add_argument("--color", type=str, default="B")
    p_ply.add_argument("--model", type=str, default=None)
    p_ply.add_argument("--simulations", type=int, default=50)
    p_ply.set_defaults(func=run_interactive)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
