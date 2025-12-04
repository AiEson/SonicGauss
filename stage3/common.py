import os
import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData

from stage2.common import (
    load_ply,
    preprocess_gaussian,
    normalize_position,
    GaussianEncoder,
    retrieve_timesteps,
)


class PositionEncoder(nn.Module):
    """Position Encoder with NeRF-style frequency encoding."""
    
    def __init__(
        self,
        num_frequencies: int = 10,
        input_dim: int = 63,
        hidden_dims: list = [256, 512],
        output_dim: int = 1024,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        
        freq_dim = 3 + 3 * 2 * num_frequencies
        assert freq_dim == input_dim, f"Expected input_dim={freq_dim}, got {input_dim}"
        
        layers = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        freqs = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer('freqs', freqs)
    
    def frequency_encoding(self, p: torch.Tensor) -> torch.Tensor:
        """Apply NeRF-style frequency encoding."""
        squeeze_batch = False
        if p.dim() == 1:
            p = p.unsqueeze(0)
            squeeze_batch = True
        
        batch_size = p.shape[0]
        p_scaled = p.unsqueeze(-1) * self.freqs.view(1, 1, -1) * math.pi
        sin_enc = torch.sin(p_scaled)
        cos_enc = torch.cos(p_scaled)
        enc = torch.stack([sin_enc, cos_enc], dim=-1)
        enc = enc.view(batch_size, 3, -1)
        enc = enc.view(batch_size, -1)
        encoded = torch.cat([enc, p], dim=-1)
        
        if squeeze_batch:
            encoded = encoded.squeeze(0)
        
        return encoded
    
    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        encoded = self.frequency_encoding(position)
        return self.mlp(encoded)


class FeatureFusion(nn.Module):
    """Feature Fusion module using Cross-Attention."""
    
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        gaussian_features: torch.Tensor,
        position_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, _ = gaussian_features.shape
        
        position_features = position_features.unsqueeze(1)
        attn_output, _ = self.cross_attn(
            query=self.norm1(gaussian_features),
            key=position_features,
            value=position_features,
        )
        attn_output = gaussian_features + attn_output
        attn_output = attn_output + self.ffn(self.norm2(attn_output))
        fused = torch.cat([gaussian_features, attn_output], dim=1)
        return fused


class Stage3Encoder(nn.Module):
    """Stage 3 Encoder combining Gaussian Encoder, Position Encoder and Feature Fusion."""
    
    def __init__(
        self,
        gaussian_encoder: GaussianEncoder,
        position_encoder: PositionEncoder,
        feature_fusion: FeatureFusion,
    ):
        super().__init__()
        self.gaussian_encoder = gaussian_encoder
        self.position_encoder = position_encoder
        self.feature_fusion = feature_fusion
    
    def forward(
        self,
        normalized_gs: dict,
        position: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward pass."""
        gaussian_features = self.gaussian_encoder(normalized_gs, device)
        if position.dim() == 1:
            position = position.unsqueeze(0)
        position_features = self.position_encoder(position)
        fused_features = self.feature_fusion(gaussian_features, position_features)
        return fused_features


def get_ply_path_from_item(item: dict, base_path: str = None) -> str:
    """Get PLY file path from dataset item."""
    ply_file = item.get('ply_file', '')
    if base_path and not os.path.isabs(ply_file):
        return os.path.join(base_path, ply_file)
    return ply_file
