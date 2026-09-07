"""PyTorch Dataset with 8-fold Dihedral Symmetry Data Augmentation for Go."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from gobot_engine.board import Move


@dataclass
class GoDataPoint:
    feature_tensor: np.ndarray  # Shape: (8, N, N)
    target_action: Union[int, np.ndarray]  # Scalar action index or distribution (N*N+1)
    target_value: float  # Value in [-1.0, 1.0]
    board_size: int = 19


class GoDataset(Dataset):
    """PyTorch Dataset providing Go state tensors and move/value targets with 8-fold symmetry."""

    def __init__(
        self,
        datapoints: Optional[List[GoDataPoint]] = None,
        augment_symmetry: bool = True,
    ):
        self.datapoints: List[GoDataPoint] = datapoints or []
        self.augment_symmetry = augment_symmetry

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dp = self.datapoints[idx]
        feat = np.copy(dp.feature_tensor)
        target_action = dp.target_action
        target_val = np.array([dp.target_value], dtype=np.float32)
        size = dp.board_size

        if self.augment_symmetry:
            feat, target_action = self._apply_random_symmetry(feat, target_action, size)

        tensor_x = torch.from_numpy(feat).float()

        if isinstance(target_action, np.ndarray):
            tensor_policy = torch.from_numpy(target_action).float()
        else:
            tensor_policy = torch.tensor(target_action, dtype=torch.long)

        tensor_val = torch.from_numpy(target_val).float()

        return tensor_x, tensor_policy, tensor_val

    def _apply_random_symmetry(
        self,
        features: np.ndarray,
        target_action: Union[int, np.ndarray],
        size: int,
    ) -> Tuple[np.ndarray, Union[int, np.ndarray]]:
        """Applies one of the 8 symmetries (4 rotations x 2 reflections)."""
        rot_k = random.randint(0, 3)
        do_flip = random.choice([True, False])

        # Rotate feature planes
        features = np.rot90(features, k=rot_k, axes=(1, 2))
        if do_flip:
            features = np.flip(features, axis=2)  # Horizontal flip

        # Transform target action
        if isinstance(target_action, int):
            if target_action >= size * size:  # Pass move remains unchanged
                new_action = target_action
            else:
                r, c = target_action // size, target_action % size
                # Create a 2D dummy grid with index
                grid = np.zeros((size, size), dtype=np.int32)
                grid[r, c] = 1
                grid = np.rot90(grid, k=rot_k)
                if do_flip:
                    grid = np.fliplr(grid)
                new_r, new_c = np.argwhere(grid == 1)[0]
                new_action = int(new_r * size + new_c)
            return np.ascontiguousarray(features), new_action
        else:
            # Distribution target (N*N + 1)
            dist = np.copy(target_action)
            board_dist = dist[:-1].reshape((size, size))
            board_dist = np.rot90(board_dist, k=rot_k)
            if do_flip:
                board_dist = np.fliplr(board_dist)
            dist[:-1] = board_dist.reshape(-1)
            return np.ascontiguousarray(features), dist

    def append(self, datapoint: GoDataPoint) -> None:
        self.datapoints.append(datapoint)

    def extend(self, datapoints: List[GoDataPoint]) -> None:
        self.datapoints.extend(datapoints)

    def clear(self) -> None:
        self.datapoints.clear()
