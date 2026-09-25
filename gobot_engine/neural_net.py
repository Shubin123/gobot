"""Dual-Head Policy and Value Deep Residual Neural Network for Go."""

from __future__ import annotations
import os
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out, inplace=True)


class GoResNet(nn.Module):
    """Dual-headed Policy and Value ResNet."""

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 8,
        num_filters: int = 64,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.action_size = board_size * board_size + 1  # Board positions + Pass

        # Initial convolutional block
        self.initial_conv = ConvBlock(in_channels, num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_blocks)])

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            x: Tensor of shape (B, in_channels, board_size, board_size)
        Returns:
            policy_logits: (B, action_size)
            value: (B, 1) in range [-1, 1]
        """
        # Ensure correct device and type
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)

        # Policy computation
        p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value computation
        v = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def predict(self, board_tensor: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, float]:
        """Inference for a single board state tensor (C, N, N).
        Returns:
            policy_probs: 1D Tensor of shape (action_size,)
            value: float in range [-1, 1]
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)
            board_tensor = board_tensor.to(device)
            logits, val = self.forward(board_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            val_scalar = float(val.squeeze(0).cpu().item())
        return probs, val_scalar

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "board_size": self.board_size,
            "in_channels": self.in_channels,
            "num_filters": self.num_filters,
            "num_blocks": self.num_blocks,
            "meta": extra_meta or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> GoResNet:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            board_size=checkpoint["board_size"],
            in_channels=checkpoint.get("in_channels", 8),
            num_filters=checkpoint.get("num_filters", 64),
            num_blocks=checkpoint.get("num_blocks", 6),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model


class DualHeadLoss(nn.Module):
    """Loss function combining CrossEntropy for Policy and MSE for Value."""

    def __init__(self, value_weight: float = 1.0):
        super().__init__()
        self.value_weight = value_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_policy_logits: torch.Tensor,
        pred_value: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes combined loss.
        Args:
            pred_policy_logits: (B, action_size)
            pred_value: (B, 1)
            target_policy: (B, action_size) probability distribution or class indices
            target_value: (B, 1) target outcome in [-1, 1]
        Returns:
            total_loss, policy_loss, value_loss
        """
        if target_policy.dim() > 1 and target_policy.size(1) == pred_policy_logits.size(1):
            # Soft target probabilities (cross-entropy with target distribution)
            log_probs = F.log_softmax(pred_policy_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(target_policy * log_probs, dim=1))
        else:
            # Discrete move index targets
            policy_loss = F.cross_entropy(pred_policy_logits, target_policy.long())

        value_loss = self.mse(pred_value, target_value)
        total_loss = policy_loss + self.value_weight * value_loss
        return total_loss, policy_loss, value_loss


# =============================================================================
# Vision Transformer Architecture for Go (AlphaZero + ViT hybrid)
# =============================================================================


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with multi-head self-attention and GELU FFN."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


class GoTransformerNet(nn.Module):
    """Vision Transformer-based dual-headed policy+value network for Go.

    Architecture inspired by AlphaZero + Vision Transformer (ViT):
    - Each board intersection is treated as a patch/token
    - Conv2d patch embedding maps 8 input channels to embed_dim
    - Learnable 2D positional encoding
    - CLS token for value head aggregation
    - Stack of pre-norm TransformerBlock layers
    - Policy head: operates on all spatial tokens
    - Value head: extracts from CLS token → scalar in [-1, 1]

    Default config: embed_dim=128, num_heads=4, num_layers=4, mlp_ratio=4.0
    Supports board sizes 9, 13, 19.
    """

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        
        self.action_size = board_size * board_size + 1
        self.num_patches = board_size * board_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Policy Head
        self.policy_conv = nn.Conv2d(embed_dim, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, E, H, W) -> (B, E, N) -> (B, N, E)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Separate CLS token and spatial tokens
        cls_token = x[:, 0]
        spatial_tokens = x[:, 1:]
        
        # Policy computation (using spatial tokens)
        spatial_2d = spatial_tokens.transpose(1, 2).reshape(B, self.embed_dim, self.board_size, self.board_size)
        p = F.relu(self.policy_bn(self.policy_conv(spatial_2d)), inplace=True)
        p = p.reshape(B, -1)
        policy_logits = self.policy_fc(p)
        
        # Value computation (using CLS token)
        value = self.value_head(cls_token)
        
        return policy_logits, value

    def predict(self, board_tensor: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, float]:
        """Inference for a single board state tensor (C, N, N).
        Returns:
            policy_probs: 1D Tensor of shape (action_size,)
            value: float in range [-1, 1]
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
            
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)
            board_tensor = board_tensor.to(device)
            logits, val = self.forward(board_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            val_scalar = float(val.squeeze(0).cpu().item())
        return probs, val_scalar

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "board_size": self.board_size,
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "meta": extra_meta or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> GoTransformerNet:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            board_size=checkpoint["board_size"],
            in_channels=checkpoint.get("in_channels", 8),
            embed_dim=checkpoint.get("embed_dim", 128),
            num_heads=checkpoint.get("num_heads", 4),
            num_layers=checkpoint.get("num_layers", 4),
            mlp_ratio=checkpoint.get("mlp_ratio", 4.0),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
