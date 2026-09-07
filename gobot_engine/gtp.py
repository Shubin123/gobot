"""Go Text Protocol (GTP 2.0) Interface Implementation."""

from __future__ import annotations
import sys
from typing import List, Optional, Tuple, Dict, Any

from .board import Board, Color, Move, PASS_MOVE, RESIGN_MOVE
from .mcts import MCTS
from .optimizer import MoveOptimizer
from .neural_net import GoResNet


class GTPEngine:
    """GTP 2.0 compliant engine bridge for Go GUI clients (Sabaki, Lizzie, Katrain)."""

    def __init__(
        self,
        board_size: int = 19,
        komi: float = 7.5,
        model: Optional[GoResNet] = None,
        num_simulations: int = 80,
    ):
        self.board = Board(size=board_size, komi=komi)
        self.model = model
        self.num_simulations = num_simulations
        self.mcts = MCTS(model=self.model, num_simulations=self.num_simulations)
        self.optimizer = MoveOptimizer(self.mcts)
        self.is_running = True

        self.commands = {
            "protocol_version": self.cmd_protocol_version,
            "name": self.cmd_name,
            "version": self.cmd_version,
            "known_command": self.cmd_known_command,
            "list_commands": self.cmd_list_commands,
            "quit": self.cmd_quit,
            "boardsize": self.cmd_boardsize,
            "clear_board": self.cmd_clear_board,
            "komi": self.cmd_komi,
            "play": self.cmd_play,
            "genmove": self.cmd_genmove,
            "showboard": self.cmd_showboard,
            "final_score": self.cmd_final_score,
            "undo": self.cmd_undo,
            "analyze": self.cmd_analyze,
        }

    def handle_command(self, line: str) -> str:
        line = line.strip()
        if not line or line.startswith("#"):
            return ""

        parts = line.split()
        cmd_id: Optional[str] = None

        if parts[0].isdigit():
            cmd_id = parts[0]
            cmd_name = parts[1].lower()
            args = parts[2:]
        else:
            cmd_name = parts[0].lower()
            args = parts[1:]

        if cmd_name not in self.commands:
            prefix = f"?{cmd_id} " if cmd_id else "? "
            return f"{prefix}unknown command: {cmd_name}\n\n"

        try:
            result = self.commands[cmd_name](args)
            prefix = f"={cmd_id} " if cmd_id else "= "
            return f"{prefix}{result}\n\n"
        except Exception as e:
            prefix = f"?{cmd_id} " if cmd_id else "? "
            return f"{prefix}{str(e)}\n\n"

    def cmd_protocol_version(self, args: List[str]) -> str:
        return "2"

    def cmd_name(self, args: List[str]) -> str:
        return "GoBot-Engine"

    def cmd_version(self, args: List[str]) -> str:
        return "1.0.0"

    def cmd_known_command(self, args: List[str]) -> str:
        if not args:
            return "false"
        return "true" if args[0].lower() in self.commands else "false"

    def cmd_list_commands(self, args: List[str]) -> str:
        return "\n".join(sorted(self.commands.keys()))

    def cmd_quit(self, args: List[str]) -> str:
        self.is_running = False
        return ""

    def cmd_boardsize(self, args: List[str]) -> str:
        if not args:
            raise ValueError("boardsize requires an argument")
        size = int(args[0])
        if size < 2 or size > 25:
            raise ValueError(f"unacceptable size: {size}")
        self.board = Board(size=size, komi=self.board.komi)
        return ""

    def cmd_clear_board(self, args: List[str]) -> str:
        self.board.reset()
        return ""

    def cmd_komi(self, args: List[str]) -> str:
        if not args:
            raise ValueError("komi requires an argument")
        self.board.komi = float(args[0])
        return ""

    def _parse_color(self, color_str: str) -> int:
        c = color_str.strip().lower()
        if c in ("b", "black"):
            return Color.BLACK
        if c in ("w", "white"):
            return Color.WHITE
        raise ValueError(f"invalid color: {color_str}")

    def cmd_play(self, args: List[str]) -> str:
        if len(args) < 2:
            raise ValueError("play requires color and vertex")
        color = self._parse_color(args[0])
        coord_str = args[1]
        move = Move.from_gtp(coord_str, self.board.size)

        if not self.board.play(move, color):
            raise ValueError(f"illegal move: {coord_str}")
        return ""

    def cmd_genmove(self, args: List[str]) -> str:
        if not args:
            raise ValueError("genmove requires color")
        color = self._parse_color(args[0])
        self.board.to_move = color

        move, meta = self.optimizer.optimize_move(self.board)
        self.board.play(move, color)
        return Move.to_gtp(move, self.board.size)

    def cmd_showboard(self, args: List[str]) -> str:
        return "\n" + self.board.render_ascii()

    def cmd_final_score(self, args: List[str]) -> str:
        res = self.board.calculate_area_score()
        return str(res["result_str"])

    def cmd_undo(self, args: List[str]) -> str:
        if not self.board.move_history:
            raise ValueError("cannot undo empty history")
        history = list(self.board.move_history)
        history.pop()  # Remove last move
        self.board.reset()
        for color, move in history:
            self.board.play(move, color)
        return ""

    def cmd_analyze(self, args: List[str]) -> str:
        candidates = self.mcts.get_top_candidates(self.board, top_k=5)
        lines = []
        for c in candidates:
            lines.append(f"move {c['gtp_coord']} visits {c['visits']} winrate {c['winrate']:.4f} prior {c['prior']:.4f}")
        return "\n".join(lines)

    def run_stdio_loop(self) -> None:
        """Standard GTP stdio loop."""
        while self.is_running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                response = self.handle_command(line)
                if response:
                    sys.stdout.write(response)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                break
