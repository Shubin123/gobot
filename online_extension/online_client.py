"""Online Bot Client and Network Adapter."""

from __future__ import annotations
import os
import sys
import time
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

# Clear inaccessible SSL keylog file
os.environ.pop("SSLKEYLOGFILE", None)

from gobot_engine.board import Board, Color, Move, PASS_MOVE
from gobot_engine.mcts import MCTS
from gobot_engine.optimizer import MoveOptimizer
from gobot_engine.neural_net import GoResNet


class OnlineBotClient:
    """Client for playing online games or connecting local engine to online servers."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        bot_color: str = "B",
        model_path: Optional[str] = None,
        num_simulations: int = 60,
    ):
        self.server_url = server_url.rstrip("/")
        self.bot_color = bot_color.upper()
        self.model = GoResNet.load_checkpoint(model_path) if model_path else None
        self.mcts = MCTS(model=self.model, num_simulations=num_simulations)
        self.optimizer = MoveOptimizer(self.mcts)

    def _http_get(self, path: str) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.server_url}{path}")
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())

    def _http_post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    def ping_server(self) -> bool:
        try:
            res = self._http_get("/api/health")
            return res.get("status") == "ok"
        except Exception:
            return False

    def fetch_state(self) -> Dict[str, Any]:
        return self._http_get("/api/board/state")

    def send_move(self, coord: str) -> Dict[str, Any]:
        return self._http_post("/api/board/play", {"color": self.bot_color, "coord": coord})

    def request_optimal_move(self, simulations: int = 60) -> Dict[str, Any]:
        return self._http_post(
            "/api/optimize/move",
            {"num_simulations": simulations, "enable_tactics": True},
        )

    def play_turn(self) -> Optional[str]:
        """Checks if it's bot's turn, computes optimal move, and plays it on the server."""
        state = self.fetch_state()
        if state.get("is_game_over"):
            return None

        if state.get("to_move") == self.bot_color:
            opt = self.request_optimal_move()
            move_coord = opt.get("selected_move", "pass")
            self.send_move(move_coord)
            return move_coord
        return None
