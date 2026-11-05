"""Shared feed-forward GQN with uncapped relation attention and synchronous messages."""

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .obs_layout import DIMS

INPUTS = (
    "self",
    "asset",
    "local_threat",
    "local_owner",
    "global_threat",
    "global_owner",
    "teammate",
    "teammate_owner",
    "teammate_src",
    "message_mask",
)


def tensors(
    graph: Mapping[str, np.ndarray], device: str | torch.device = "cpu"
) -> tuple[Tensor, ...]:
    """Convert packed C++ features and indices to model inputs, without changing their meaning."""
    return tuple(
        torch.as_tensor(np.array(graph[k], copy=True), device=device)
        if k != "message_mask"
        else torch.ones(len(graph["self"]), device=device)
        for k in INPUTS
    )


def mlp(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU(), nn.Linear(n_out, n_out), nn.ReLU())


class Relation(nn.Module):
    """Four-head attention over nodes grouped by owning agent, with a learned empty value."""

    def __init__(self, features: int, hidden: int, heads: int):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.encoder = mlp(features, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.empty = nn.Parameter(torch.zeros(hidden))

    def forward(self, query: Tensor, nodes: Tensor, owner: Tensor) -> tuple[Tensor, Tensor]:
        # A single internal sentinel agent/node keeps ONNX kernels well-defined for
        # empty relations. It is isolated from real agents and removed on output.
        query = torch.cat((query.new_zeros((1, query.shape[1])), query), 0)
        nodes = torch.cat((nodes.new_zeros((1, nodes.shape[1])), nodes), 0)
        owner = torch.cat((owner.new_zeros(1), owner + 1), 0)
        encoded = self.encoder(nodes)
        keys = self.key(encoded).reshape(-1, self.heads, self.width)
        values = self.value(encoded).reshape(-1, self.heads, self.width)
        queries = self.query(query).reshape(-1, self.heads, self.width)
        logits = (queries[owner] * keys).sum(-1) / self.width**0.5
        # Bounded logits keep the grouped exponential sum stable and export to portable ONNX ops.
        weights = logits.clamp(-20, 20).exp()
        denom = query.new_zeros((query.shape[0], self.heads)).scatter_add(
            0, owner[:, None].expand_as(weights), weights
        )
        attention = weights / denom[owner].clamp_min(1e-12)
        weighted = attention.unsqueeze(-1) * values
        pooled = (
            query.new_zeros((query.shape[0], self.heads, self.width))
            .scatter_add(0, owner[:, None, None].expand_as(weighted), weighted)
            .flatten(1)
        )
        return torch.where(denom.sum(-1, keepdim=True) > 0, pooled, self.empty)[1:], attention[1:]


class GQN(nn.Module):
    """Local encoding → one message round → global encoding → 27 Q-values per alive agent."""

    def __init__(self, hidden: int = 32, heads: int = 4, message_dim: int = 16):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden size must be divisible by heads")
        self.hidden, self.heads, self.message_dim = hidden, heads, message_dim
        self.own = mlp(DIMS["self"], hidden)
        self.asset = mlp(DIMS["asset"], hidden)
        self.lt = Relation(DIMS["local_threat"], hidden, heads)
        self.gt = Relation(DIMS["global_threat"], hidden, heads)
        self.tm = Relation(DIMS["teammate"] + message_dim, hidden, heads)
        self.local = mlp(hidden * 3, hidden)
        self.message = nn.Linear(hidden, message_dim)
        self.global_ = mlp(hidden * 2, hidden)
        self.q = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.ReLU(), nn.Linear(hidden, 27))

    def forward(
        self,
        own: Tensor,
        asset: Tensor,
        lt: Tensor,
        lo: Tensor,
        gt: Tensor,
        go: Tensor,
        tm: Tensor,
        to: Tensor,
        ts: Tensor,
        message_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.own(own)
        z_lt, a_lt = self.lt(h, lt, lo)
        local = self.local(torch.cat((h, z_lt, self.asset(asset)), -1))
        messages = self.message(local) * message_mask.unsqueeze(-1)
        z_gt, a_gt = self.gt(h, gt, go)
        z_tm, a_tm = self.tm(h, torch.cat((tm, messages[ts]), -1), to)
        g = own[:, 8:9]
        global_ = g * self.global_(torch.cat((z_gt, z_tm), -1))
        q = self.q(torch.cat((local, global_, g), -1))
        return q, a_lt, a_gt, a_tm, messages

    def settings(self) -> dict[str, int]:
        """Constructor settings stored alongside checkpoints."""
        return {"hidden": self.hidden, "heads": self.heads, "message_dim": self.message_dim}
