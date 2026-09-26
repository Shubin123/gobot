"""Command Line Interface for GoBot Engine, Training, GTP, and Server."""

import os
import sys
import argparse
import uvicorn

from gobot_engine.board import Board, Color, Move, PASS_MOVE, RESIGN_MOVE
from gobot_engine.neural_net import GoResNet, GoTransformerNet
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


def _load_model_auto(path, model_type="auto", board_size=9):
    """Loads a model from checkpoint, auto-detecting type, or creates a new one."""
    if path and os.path.exists(path):
        # Try loading and detecting model type from checkpoint
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "embed_dim" in checkpoint or model_type == "transformer":
            return GoTransformerNet.load_checkpoint(path)
        return GoResNet.load_checkpoint(path)

    # Create new model
    if model_type == "transformer":
        return GoTransformerNet(board_size=board_size)
    return None


def run_train_rl(args):
    """Runs the deep RL training loop (AlphaZero-style self-play + training + gating)."""
    from training_pipeline.rl_trainer import RLTrainer

    print(f"\n{'='*60}")
    print(f"  GoBot Deep RL Training (AlphaZero-style)")
    print(f"  Model: {args.model_type} | Board: {args.board_size}x{args.board_size}")
    print(f"  Iterations: {args.iterations} | Games/iter: {args.games_per_iter}")
    print(f"{'='*60}\n")

    model = _load_model_auto(args.init_model, args.model_type, args.board_size)

    if model is None:
        if args.model_type == "transformer":
            model = GoTransformerNet(board_size=args.board_size)
        else:
            model = GoResNet(board_size=args.board_size, num_filters=args.filters, num_blocks=args.blocks)

    rl_trainer = RLTrainer(
        model=model,
        board_size=args.board_size,
        num_simulations=args.simulations,
        lr=args.lr,
    )

    rl_trainer.run_rl_training(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        training_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_games=args.eval_games,
        win_threshold=args.win_threshold,
        checkpoint_dir=args.checkpoint_dir,
        output_model_name=args.output_model,
    )


def run_train_pro_data(args):
    """Trains on real professional game data downloaded from public archives."""
    from training_pipeline.pro_games import build_comprehensive_pro_dataset

    print(f"\n{'='*60}")
    print(f"  GoBot Pro Game Training (REAL DATA)")
    print(f"  Model: {args.model_type} | Board: {args.board_size}x{args.board_size}")
    print(f"  Source: {args.source} | Max games: {args.max_games or 'all'}")
    print(f"{'='*60}\n")

    # Build dataset from real downloaded games
    sgf_dirs = args.sgf_dirs if hasattr(args, 'sgf_dirs') and args.sgf_dirs else None
    dataset = build_comprehensive_pro_dataset(
        board_size=args.board_size,
        max_games=args.max_games,
        data_dir=args.data_dir,
        source=args.source,
        sgf_dirs=sgf_dirs,
    )
    print(f"Loaded {len(dataset)} training positions from professional games.")

    if len(dataset) == 0:
        print("Error: No training positions found.", file=sys.stderr)
        print("  Try: gobot download-games --source cwi", file=sys.stderr)
        return

    model = _load_model_auto(args.init_model, args.model_type, args.board_size)
    if model is None:
        if args.model_type == "transformer":
            model = GoTransformerNet(board_size=args.board_size)
        else:
            model = GoResNet(board_size=args.board_size, num_filters=args.filters, num_blocks=args.blocks)

    trainer = GoTrainer(model, lr=args.lr)
    history = trainer.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.output_model,
    )

    print(f"\nTraining complete. Model saved to {os.path.join(args.checkpoint_dir, args.output_model)}")
    for h in history:
        print(f"Epoch {h['epoch']}: Loss={h['loss']:.4f} | Top1 Acc={h['top1_accuracy']*100:.1f}%")


def run_download_games(args):
    """Downloads real professional game archives for training."""
    from training_pipeline.sgf_downloader import download_and_prepare, list_available_sources

    if args.list_sources:
        list_available_sources()
        return

    print(f"\n  Downloading {args.source} game archive...")
    files = download_and_prepare(
        source_key=args.source,
        output_dir=args.data_dir,
        board_size=args.board_size if hasattr(args, 'board_size') else None,
        force=args.force,
    )
    print(f"\n  Done! {len(files)} SGF files ready for training.")
    print(f"  Use: gobot train-pro-data --source {args.source} --data-dir {args.data_dir}")


def run_benchmark(args):
    from benchmark_suite import BenchmarkSuite, print_benchmark_report
    model = _load_model_auto(args.model, board_size=args.board_size)
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

    # Train-RL (Deep Reinforcement Learning — AlphaZero-style self-play + training)
    p_rl = subparsers.add_parser("train-rl", help="Run deep RL training (AlphaZero-style self-play)")
    p_rl.add_argument("--board-size", type=int, default=9)
    p_rl.add_argument("--model-type", type=str, default="resnet", choices=["resnet", "transformer"],
                       help="Neural network architecture (resnet or transformer)")
    p_rl.add_argument("--iterations", type=int, default=10, help="Number of RL iterations")
    p_rl.add_argument("--games-per-iter", type=int, default=20, help="Self-play games per iteration")
    p_rl.add_argument("--epochs", type=int, default=5, help="Training epochs per iteration")
    p_rl.add_argument("--batch-size", type=int, default=32)
    p_rl.add_argument("--lr", type=float, default=1e-3)
    p_rl.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    p_rl.add_argument("--eval-games", type=int, default=10, help="Evaluation games for gating")
    p_rl.add_argument("--win-threshold", type=float, default=0.55, help="Win rate to accept new model")
    p_rl.add_argument("--blocks", type=int, default=4)
    p_rl.add_argument("--filters", type=int, default=48)
    p_rl.add_argument("--init-model", type=str, default=None, help="Initial model checkpoint")
    p_rl.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_rl.add_argument("--output-model", type=str, default="rl_gobot_model.pt")
    p_rl.set_defaults(func=run_train_rl)

    # Train-Pro-Data (Supervised training on REAL professional game data)
    p_pd = subparsers.add_parser("train-pro-data", help="Train on real professional game archives (downloads 90K+ games)")
    p_pd.add_argument("--board-size", type=int, default=19, help="Board size to filter games (9, 13, or 19)")
    p_pd.add_argument("--model-type", type=str, default="resnet", choices=["resnet", "transformer"],
                       help="Neural network architecture (resnet or transformer)")
    p_pd.add_argument("--source", type=str, default="cwi", choices=["cwi", "jgdb"],
                       help="Data source: cwi (90K games) or jgdb (500K games)")
    p_pd.add_argument("--max-games", type=int, default=None, help="Limit number of games to process")
    p_pd.add_argument("--data-dir", type=str, default="sgf_data", help="Directory to cache downloaded SGF archives")
    p_pd.add_argument("--sgf-dirs", nargs="+", default=None, help="Custom SGF directories (skip download)")
    p_pd.add_argument("--epochs", type=int, default=10)
    p_pd.add_argument("--batch-size", type=int, default=64)
    p_pd.add_argument("--lr", type=float, default=1e-3)
    p_pd.add_argument("--blocks", type=int, default=6)
    p_pd.add_argument("--filters", type=int, default=64)
    p_pd.add_argument("--init-model", type=str, default=None, help="Initial model checkpoint")
    p_pd.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_pd.add_argument("--output-model", type=str, default="pro_trained_model.pt")
    p_pd.set_defaults(func=run_train_pro_data)

    # Download-Games (Download real game archives)
    p_dl = subparsers.add_parser("download-games", help="Download real professional game archives for training")
    p_dl.add_argument("--source", type=str, default="cwi", choices=["cwi", "jgdb"],
                       help="cwi = 90K Japanese pro games (46MB), jgdb = 500K games (194MB)")
    p_dl.add_argument("--data-dir", type=str, default="sgf_data", help="Where to store downloaded games")
    p_dl.add_argument("--board-size", type=int, default=None, help="Filter for specific board size")
    p_dl.add_argument("--force", action="store_true", help="Re-download even if already exists")
    p_dl.add_argument("--list-sources", action="store_true", help="List all available data sources")
    p_dl.set_defaults(func=run_download_games)

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
