"""Per-organ learned attention pool over Pillar-0 token sets.

Drop-in alternative to the hand-coded mean/max/topk pools. Each organ has a learned
query vector that cross-attends to the K cached tokens within the organ mask. Output
is one 384-d feature vector per organ -- same shape as `mean` or `max` -- so it can
be concatenated alongside them as additional node features for the GNN.

Trains jointly with the downstream R-GAT on finding labels, so the attention learns
*what to look for* per organ given the supervision -- something a fixed heuristic
(mean / max / top-k) can't do.

    pool = OrganAttentionPool(num_organs=16, dim=384, num_heads=4)
    feats = pool(tokens, organ_idx, key_padding_mask=padded_mask)
    #  tokens (N, K, C) -> feats (N, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OrganAttentionPool(nn.Module):
    def __init__(self, num_organs: int, dim: int = 384, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        # per-organ learned query (small init: 0.02 std, standard transformer practice)
        self.queries = nn.Parameter(torch.randn(num_organs, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        # final per-dim norm for stability when concat'd with mean/max
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, organ_idx: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        tokens: (N, K, C)         — N nodes per batch, K tokens, C dim
        organ_idx: (N,)           — which organ each node represents (looks up its query)
        key_padding_mask: (N, K)  — True = padded token (ignored by attention)
        Returns: (N, C)           — one pooled feature vector per node
        """
        q = self.queries[organ_idx].unsqueeze(1)              # (N, 1, C)
        pooled, _ = self.attn(q, tokens, tokens,
                              key_padding_mask=key_padding_mask, need_weights=False)
        return self.norm(pooled.squeeze(1))                   # (N, C)
