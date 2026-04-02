# -*- coding: utf-8 -*-
# ===============================
# models.py
# ===============================
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from utils import WIN_SIZE, compute_tfr


class TransformerEncoder(nn.Module):
    """标准 Transformer Encoder 堆叠。"""
    def __init__(self, d_model: int = 384, nhead: int = 6,
                 num_layers: int = 6, dim_feedforward: int = 1536,
                 dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        return self.encoder(x)


class EEGSSLModel(nn.Module):
    def __init__(self,
                 d_model: int = 384,
                 nhead: int = 4,
                 num_layers: int = 11,
                 dim_feedforward: int = 1536,
                 dropout: float = 0.1):
        super().__init__()

        dummy = np.zeros((3, WIN_SIZE), dtype=np.float32)
        spec_example = compute_tfr(dummy)  
        self.seq_len = int(spec_example.shape[0])
        self.in_dim = int(spec_example.shape[1])

        self.d_model = d_model

        self.input_proj = nn.Linear(self.in_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.seq_len, d_model))

        self.backbone = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.recon_head = nn.Linear(d_model, self.in_dim)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def encode(self, spec: torch.Tensor) -> torch.Tensor:
        B, T, D = spec.shape
        assert T == self.seq_len, f"时频序列长度不一致: got {T}, expect {self.seq_len}"
        assert D == self.in_dim, f"特征维度不一致: got {D}, expect {self.in_dim}"

        x = self.input_proj(spec)
        cls = self.cls_token.expand(B, 1, -1)
        h = torch.cat([cls, x], dim=1)
        h = h + self.pos_emb[:, : h.shape[1], :]
        h = self.backbone(h)
        return h

    def reconstruct(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_h = h[:, 1:, :]
        masked_h = token_h[mask]
        rec = self.recon_head(masked_h)
        return rec


class EEGClassifier(nn.Module):
    def __init__(
        self,
        ssl_model: EEGSSLModel,
        num_classes: int = 4,
        head_type: str = "mlp",
        norm_type: Optional[str] = "layernorm",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = ssl_model.input_proj
        self.cls_token  = ssl_model.cls_token
        self.pos_emb    = ssl_model.pos_emb
        self.backbone   = ssl_model.backbone
        self.seq_len    = ssl_model.seq_len
        self.in_dim     = ssl_model.in_dim

        d_model = self.pos_emb.shape[-1]
        if hidden_dim is None:
            hidden_dim = d_model

        hidden_dim2 = max(1, hidden_dim // 2)

        norm_layer: nn.Module = nn.Identity()
        if norm_type is not None:
            if norm_type.lower() == "layernorm":
                norm_layer = nn.LayerNorm(d_model)
            elif norm_type.lower() == "batchnorm":
                norm_layer = nn.BatchNorm1d(d_model)
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}")

        head_type = head_type.lower()
        if head_type == "linear":
            self.head = nn.Sequential(
                norm_layer,
                nn.Linear(d_model, num_classes),
            )
        elif head_type == "mlp":
            self.head = nn.Sequential(
                norm_layer,
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim2, num_classes),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

        self.head_type = head_type
        self.norm_type = norm_type
        self.d_model   = d_model
        self.num_classes = num_classes

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        B, T, D = spec.shape
        assert T == self.seq_len, f"时频序列长度不一致: got {T}, expect {self.seq_len}"
        assert D == self.in_dim, f"特征维度不一致: got {D}, expect {self.in_dim}"

        x = self.input_proj(spec)
        cls = self.cls_token.expand(B, 1, -1)
        h = torch.cat([cls, x], dim=1)
        h = h + self.pos_emb[:, : h.shape[1], :]
        h = self.backbone(h)
        cls_h = h[:, 0, :]

        logits = self.head(cls_h)
        return logits
