"""Monte Carlo Tree Search (MCTS) with PUCT and Neural Net Guidance."""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from .board import Board, Color, Move, PASS_MOVE, RESIGN_MOVE
from .neural_net import GoResNet


class MCTSNode:
    """A node in the Monte Carlo Tree Search."""

    def __init__(self, prior: float = 0.0, parent: Optional[MCTSNode] = None):
        self.prior: float = prior
        self.parent: Optional[MCTSNode] = parent
        self.children: Dict[int, MCTSNode] = {}  # action_index -> MCTSNode
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        u = c_puct * self.prior * (math.sqrt(parent_visits) / (1 + self.visit_count))
        return self.q_value + u


class MCTS:
    """MCTS Engine with PUCT formula and optional Neural Network evaluation."""

    def __init__(
        self,
        model: Optional[GoResNet] = None,
        c_puct: float = 2.0,
        num_simulations: int = 100,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        device: str = "cpu",
    ):
        self.model: Optional[GoResNet] = model
        self.c_puct: float = c_puct
        self.num_simulations: int = num_simulations
        self.dirichlet_alpha: float = dirichlet_alpha
        self.dirichlet_epsilon: float = dirichlet_epsilon
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        if self.model:
            self.model.to(self.device)
            self.model.eval()

    def search(
        self,
        board: Board,
        temperature: float = 0.0,
        add_noise: bool = False,
    ) -> Tuple[Tuple[int, int], np.ndarray, float]:
        """Runs MCTS search from the current board position.
        Returns:
            best_move: (r, c) or PASS_MOVE
            action_probs: 1D probability distribution over all actions (N*N + 1)
            root_value: Estimated winrate from current player's perspective in [-1, 1]
        """
        root = MCTSNode()
        self._expand_and_evaluate(root, board)

        if add_noise and root.children:
            # Add Dirichlet noise to root priors for exploration
            actions = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            for i, act in enumerate(actions):
                root.children[act].prior = (
                    (1 - self.dirichlet_epsilon) * root.children[act].prior
                    + self.dirichlet_epsilon * noise[i]
                )

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_board = board.clone()
            search_path: List[Tuple[MCTSNode, int]] = []

            # 1. Selection
            while node.is_expanded and not sim_board.is_game_over:
                if not node.children:
                    break
                best_action, next_node = self._select_best_child(node)
                search_path.append((node, best_action))
                move = Move.from_action_index(best_action, sim_board.size)
                sim_board.play(move)
                node = next_node

            # 2. Evaluation & Expansion
            if sim_board.is_game_over:
                score_dict = sim_board.calculate_area_score()
                winner = score_dict["winner"]
                # Value from perspective of current player in sim_board
                if winner == sim_board.to_move:
                    val = 1.0
                elif winner == Color.opponent(sim_board.to_move):
                    val = -1.0
                else:
                    val = 0.0
            else:
                val = self._expand_and_evaluate(node, sim_board)

            # 3. Backpropagation (flipping perspective at each alternating step)
            # The evaluated val is from perspective of sim_board.to_move
            curr_val = val
            for p_node, action in reversed(search_path):
                # Opponent moved to reach this state, so invert value
                curr_val = -curr_val
                p_node.visit_count += 1
                p_node.value_sum += curr_val

        # Construct action probabilities from visit counts
        action_size = board.size * board.size + 1
        visits = np.zeros(action_size, dtype=np.float32)
        for act, child in root.children.items():
            visits[act] = child.visit_count

        if visits.sum() == 0:
            action_probs = np.zeros(action_size, dtype=np.float32)
            action_probs[-1] = 1.0
            return PASS_MOVE, action_probs, 0.0

        if temperature == 0.0:
            # Deterministic / Greedy choice
            best_action = int(np.argmax(visits))
            action_probs = np.zeros(action_size, dtype=np.float32)
            action_probs[best_action] = 1.0
        else:
            # Softmax / temperature scaling
            visits_temp = visits ** (1.0 / temperature)
            total_v = visits_temp.sum()
            action_probs = visits_temp / (total_v if total_v > 0 else 1.0)
            best_action = int(np.random.choice(len(action_probs), p=action_probs))

        best_move = Move.from_action_index(best_action, board.size)
        root_value = root.q_value

        return best_move, action_probs, root_value

    def _select_best_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        best_score = -float("inf")
        best_action = -1
        best_child = None

        parent_visits = max(1, node.visit_count)
        for act, child in node.children.items():
            score = child.ucb_score(parent_visits, self.c_puct)
            if score > best_score:
                best_score = score
                best_action = act
                best_child = child

        return best_action, best_child  # type: ignore

    def _expand_and_evaluate(self, node: MCTSNode, board: Board) -> float:
        """Expands node with legal moves and returns evaluation from perspective of board.to_move."""
        node.is_expanded = True
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return 0.0

        action_size = board.size * board.size + 1
        legal_action_indices = [Move.to_action_index(m, board.size) for m in legal_moves]

        if self.model is not None and getattr(self.model, "board_size", None) == board.size:
            tensor = board.to_feature_tensor(board.to_move)
            policy_probs, value = self.model.predict(tensor, device=self.device)
            policy_np = policy_probs.numpy()
        else:
            # Fast heuristic / uniform priors
            policy_np = np.ones(action_size, dtype=np.float32) / action_size
            value = 0.0

        # Mask out illegal moves and renormalize
        mask = np.zeros(action_size, dtype=np.float32)
        for idx in legal_action_indices:
            mask[idx] = 1.0

        masked_priors = policy_np * mask
        sum_priors = masked_priors.sum()
        if sum_priors > 0:
            masked_priors /= sum_priors
        else:
            for idx in legal_action_indices:
                masked_priors[idx] = 1.0 / len(legal_action_indices)

        for idx in legal_action_indices:
            node.children[idx] = MCTSNode(prior=float(masked_priors[idx]), parent=node)

        return value

    def get_top_candidates(self, board: Board, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns analysis of top candidate moves with visits, winrate, and prior."""
        root = MCTSNode()
        self._expand_and_evaluate(root, board)

        for _ in range(self.num_simulations):
            node = root
            sim_board = board.clone()
            search_path = []

            while node.is_expanded and not sim_board.is_game_over:
                if not node.children:
                    break
                best_action, next_node = self._select_best_child(node)
                search_path.append((node, best_action))
                move = Move.from_action_index(best_action, sim_board.size)
                sim_board.play(move)
                node = next_node

            if sim_board.is_game_over:
                score_dict = sim_board.calculate_area_score()
                winner = score_dict["winner"]
                val = 1.0 if winner == sim_board.to_move else (-1.0 if winner == Color.opponent(sim_board.to_move) else 0.0)
            else:
                val = self._expand_and_evaluate(node, sim_board)

            curr_val = val
            for p_node, _ in reversed(search_path):
                curr_val = -curr_val
                p_node.visit_count += 1
                p_node.value_sum += curr_val

        candidates = []
        for act, child in root.children.items():
            move = Move.from_action_index(act, board.size)
            # Winrate converted from [-1, 1] to [0.0, 1.0] (win probability for current player)
            # child.q_value is from opponent's perspective when child was traversed, so current perspective is -child.q_value
            winrate = (-child.q_value + 1.0) / 2.0 if child.visit_count > 0 else 0.5
            candidates.append({
                "action_index": act,
                "move": move,
                "gtp_coord": Move.to_gtp(move, board.size),
                "visits": child.visit_count,
                "prior": round(child.prior, 4),
                "winrate": round(winrate, 4),
            })

        candidates.sort(key=lambda x: (x["visits"], x["winrate"]), reverse=True)
        return candidates[:top_k]
