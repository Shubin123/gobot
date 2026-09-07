"""FastAPI REST & WebSocket Server for Online Game Engine & Extension."""

from __future__ import annotations
import os
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from gobot_engine.board import Board, Color, Move, PASS_MOVE, RESIGN_MOVE
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from gobot_engine.neural_net import GoResNet


class PlayMoveRequest(BaseModel):
    color: str = Field(..., description="'B' or 'W'")
    coord: str = Field(..., description="GTP coordinate e.g. 'D4', 'pass', 'resign'")


class ResetBoardRequest(BaseModel):
    board_size: int = Field(19, ge=5, le=25)
    komi: float = 7.5


class OptimizeMoveRequest(BaseModel):
    board_size: Optional[int] = 19
    komi: Optional[float] = 7.5
    grid: Optional[List[List[int]]] = None  # Optional custom grid: 0=Empty, 1=Black, 2=White
    to_move: Optional[str] = "B"
    num_simulations: Optional[int] = 60
    enable_tactics: Optional[bool] = True


def create_app(model_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="GoBot Engine & Online Extension API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auto-discover winning model if not specified
    if not model_path:
        for candidate in [
            "checkpoints/winning_gobot_model.pt",
            "checkpoints/gobot_model.pt",
            "checkpoints/demo_model.pt",
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break

    # Initialize neural model if checkpoint exists
    model: Optional[GoResNet] = None
    if model_path and os.path.exists(model_path):
        try:
            model = GoResNet.load_checkpoint(model_path)
        except Exception:
            model = None

    # Load optimal hyperparameters if present
    c_puct = 2.0
    num_sims = 60
    opt_json = "checkpoints/optimal_hyperparams.json"
    if os.path.exists(opt_json):
        try:
            import json
            with open(opt_json, "r") as f:
                opt_data = json.load(f)
                best_h = opt_data.get("best_hyperparameters", {})
                c_puct = best_h.get("c_puct", c_puct)
                num_sims = best_h.get("num_simulations", num_sims)
        except Exception:
            pass

    # Global active board state
    active_board = Board(size=19, komi=7.5)
    mcts_engine = MCTS(model=model, c_puct=c_puct, num_simulations=num_sims)
    optimizer = MoveOptimizer(mcts=mcts_engine, enable_tactics=True, use_joseki_book=True)

    @app.get("/api/health")
    def health_check():
        return {"status": "ok", "board_size": active_board.size, "has_model": model is not None}

    @app.post("/api/board/reset")
    def reset_board(req: ResetBoardRequest):
        nonlocal active_board
        active_board = Board(size=req.board_size, komi=req.komi)
        return {
            "status": "reset",
            "board_size": active_board.size,
            "komi": active_board.komi,
            "to_move": "B" if active_board.to_move == Color.BLACK else "W",
        }

    @app.post("/api/board/play")
    def play_move(req: PlayMoveRequest):
        color = Color.BLACK if req.color.upper() == "B" else Color.WHITE
        try:
            move = Move.from_gtp(req.coord, active_board.size)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        success = active_board.play(move, color)
        if not success:
            raise HTTPException(status_code=400, detail=f"Illegal move: {req.coord}")

        score = active_board.calculate_area_score()
        return {
            "status": "played",
            "move": req.coord,
            "color": req.color.upper(),
            "to_move": "B" if active_board.to_move == Color.BLACK else "W",
            "is_game_over": active_board.is_game_over,
            "score": score["result_str"],
        }

    @app.get("/api/board/state")
    def get_board_state():
        score = active_board.calculate_area_score()
        return {
            "board_size": active_board.size,
            "komi": active_board.komi,
            "grid": active_board.grid.tolist(),
            "to_move": "B" if active_board.to_move == Color.BLACK else "W",
            "captures": {
                "black": active_board.captures[Color.BLACK],
                "white": active_board.captures[Color.WHITE],
            },
            "move_history_length": len(active_board.move_history),
            "is_game_over": active_board.is_game_over,
            "score": score["result_str"],
        }

    @app.post("/api/optimize/move")
    def optimize_move_endpoint(req: OptimizeMoveRequest):
        # Use custom board if grid supplied, else use active_board
        if req.grid is not None:
            size = len(req.grid)
            eval_board = Board(size=size, komi=req.komi or 7.5)
            import numpy as np
            eval_board.grid = np.array(req.grid, dtype=np.int8)
            eval_board.to_move = Color.BLACK if (req.to_move or "B").upper() == "B" else Color.WHITE
        else:
            eval_board = active_board.clone()

        local_mcts = MCTS(model=model, num_simulations=req.num_simulations or 60)
        local_optimizer = MoveOptimizer(mcts=local_mcts, enable_tactics=req.enable_tactics if req.enable_tactics is not None else True)

        best_move, metadata = local_optimizer.optimize_move(eval_board)

        return {
            "selected_move": metadata["gtp_coord"],
            "row": best_move[0] if best_move[0] >= 0 else None,
            "col": best_move[1] if best_move[1] >= 0 else None,
            "winrate": metadata["winrate"],
            "tactical_override": metadata["tactical_override"],
            "top_candidates": metadata["top_candidates"],
        }

    @app.post("/api/analyze")
    def analyze_endpoint(req: OptimizeMoveRequest):
        if req.grid is not None:
            import numpy as np
            size = len(req.grid)
            eval_board = Board(size=size, komi=req.komi or 7.5)
            eval_board.grid = np.array(req.grid, dtype=np.int8)
            eval_board.to_move = Color.BLACK if (req.to_move or "B").upper() == "B" else Color.WHITE
        else:
            eval_board = active_board.clone()

        local_mcts = MCTS(model=model, num_simulations=req.num_simulations or 60)
        candidates = local_mcts.get_top_candidates(eval_board, top_k=8)
        score_dict = eval_board.calculate_area_score()

        return {
            "current_score_estimate": score_dict["result_str"],
            "black_territory_count": score_dict["black_score"],
            "white_territory_count": score_dict["white_score"],
            "candidates": candidates,
        }

    @app.websocket("/ws/game")
    async def websocket_game(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")
                if action == "get_state":
                    await websocket.send_json(get_board_state())
                elif action == "play":
                    color_str = data.get("color", "B")
                    coord_str = data.get("coord", "pass")
                    color = Color.BLACK if color_str.upper() == "B" else Color.WHITE
                    move = Move.from_gtp(coord_str, active_board.size)
                    active_board.play(move, color)
                    await websocket.send_json(get_board_state())
                elif action == "genmove":
                    color_str = data.get("color", "B")
                    color = Color.BLACK if color_str.upper() == "B" else Color.WHITE
                    active_board.to_move = color
                    best_move, meta = optimizer.optimize_move(active_board)
                    active_board.play(best_move, color)
                    resp = get_board_state()
                    resp["bot_move"] = meta["gtp_coord"]
                    resp["winrate"] = meta["winrate"]
                    await websocket.send_json(resp)
        except WebSocketDisconnect:
            pass

    # Static web UI files
    web_dir = os.path.join(os.path.dirname(__file__), "web_ui")
    if os.path.exists(web_dir):
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index_page():
            index_path = os.path.join(web_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return "<html><body><h2>GoBot Engine Active</h2></body></html>"

    return app


app = create_app()
