# -*- coding: utf-8 -*-
# ===============================
# models.py —— “时频 token + 掩码重建” 自监督主干 + 分类器
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
    """
    自监督模型（BrainBERT 风格，只做掩码重建）：

      输入:  每个样本是一段 4s EEG 的时频图，shape = (T_frames, D_feat)
             —— 行是时间帧 token，列是频率×通道拼接后的特征维；
      主干:  Linear(input_dim→d_model) + 可学习 pos_emb + [CLS] + Transformer；
      任务:  随机掩码一部分时间帧 token，重建被掩码 token 对应的 D_feat 向量。
    """
    def __init__(self,
                 d_model: int = 384,
                 nhead: int = 4,
                 num_layers: int = 11,
                 dim_feedforward: int = 1536,
                 dropout: float = 0.1):
        super().__init__()

        # 利用一个 dummy 4s 窗，推断时频图的 (T_frames, D_feat)
        dummy = np.zeros((3, WIN_SIZE), dtype=np.float32)
        spec_example = compute_tfr(dummy)  # (T_frames, D_feat)
        self.seq_len = int(spec_example.shape[0])
        self.in_dim = int(spec_example.shape[1])

        self.d_model = d_model

        # 时频 token 投影到 d_model
        self.input_proj = nn.Linear(self.in_dim, d_model)

        # [CLS] token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.seq_len, d_model))

        # Transformer 主干
        self.backbone = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # 重建头：d_model → D_feat
        self.recon_head = nn.Linear(d_model, self.in_dim)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def encode(self, spec: torch.Tensor) -> torch.Tensor:
        """
        编码时频 token 序列。
        spec: (B, T_frames, D_feat)
        返回: (B, 1+T_frames, d_model)，包含 CLS。
        """
        B, T, D = spec.shape
        assert T == self.seq_len, f"时频序列长度不一致: got {T}, expect {self.seq_len}"
        assert D == self.in_dim, f"特征维度不一致: got {D}, expect {self.in_dim}"

        x = self.input_proj(spec)                 # (B,T,d_model)
        cls = self.cls_token.expand(B, 1, -1)     # (B,1,d_model)
        h = torch.cat([cls, x], dim=1)            # (B,1+T,d_model)
        h = h + self.pos_emb[:, : h.shape[1], :]  # 加位置编码
        h = self.backbone(h)                      # (B,1+T,d_model)
        return h

    def reconstruct(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        仅对被掩码的时间帧进行重建。
        mask: (B, T_frames) bool
        返回:
            rec: (num_masked, D_feat)
        """
        # 去掉 CLS，只保留时间帧 token
        token_h = h[:, 1:, :]    # (B,T,d_model)
        masked_h = token_h[mask] # (#masked,d_model)
        rec = self.recon_head(masked_h)  # (#masked, D_feat)
        return rec


class EEGClassifier(nn.Module):
    """
    复用自监督主干（input_proj + pos_emb + backbone），
    使用 CLS 向量 + 线性 / MLP 头做分类。
    """
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
        # 复用自监督主干参数
        self.input_proj = ssl_model.input_proj
        self.cls_token  = ssl_model.cls_token
        self.pos_emb    = ssl_model.pos_emb
        self.backbone   = ssl_model.backbone
        self.seq_len    = ssl_model.seq_len
        self.in_dim     = ssl_model.in_dim

        d_model = self.pos_emb.shape[-1]
        if hidden_dim is None:
            hidden_dim = d_model

        # 新增：第二隐层宽度（默认为一半）
        hidden_dim2 = max(1, hidden_dim // 2)

        # 归一化选择
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
            # 三层 Linear：d_model -> hidden_dim -> hidden_dim2 -> num_classes
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
        """
        spec: (B, T_frames, D_feat)
        return: (B, num_classes)
        """
        B, T, D = spec.shape
        assert T == self.seq_len, f"时频序列长度不一致: got {T}, expect {self.seq_len}"
        assert D == self.in_dim, f"特征维度不一致: got {D}, expect {self.in_dim}"

        x = self.input_proj(spec)                # (B,T,d_model)
        cls = self.cls_token.expand(B, 1, -1)    # (B,1,d_model)
        h = torch.cat([cls, x], dim=1)           # (B,1+T,d_model)
        h = h + self.pos_emb[:, : h.shape[1], :] # 位置编码
        h = self.backbone(h)                     # (B,1+T,d_model)
        cls_h = h[:, 0, :]                       # (B,d_model)

        logits = self.head(cls_h)                # (B,num_classes)
        return logits
