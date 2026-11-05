"""Packed team graphs, uniform ring replay and episode-aware n-step returns."""

import pickle
import zlib
from collections import deque
from dataclasses import dataclass

import numpy as np

Graph = dict[str, np.ndarray]
OWNERS = {
    "local_threat": "local_owner",
    "global_threat": "global_owner",
    "teammate": "teammate_owner",
}
INDICES = ("local_owner", "global_owner", "teammate_owner", "teammate_src")


def split(graph: Graph, env: int) -> Graph:
    """Extract one team and remap node owners and teammate sources to its local agent indices."""
    agents = np.flatnonzero(graph["agent_env"] == env)
    remap = np.full(len(graph["self"]), -1, np.int64)
    remap[agents] = np.arange(len(agents))
    result = {
        k: np.array(graph[k][agents], copy=True)
        for k in ("self", "asset", "agent_env", "agent_drone", "alive_mask")
    }
    result["agent_env"][:] = 0
    for kind, key in OWNERS.items():
        keep = np.isin(graph[key], agents)
        result[kind] = np.array(graph[kind][keep], copy=True)
        result[key] = remap[graph[key][keep]]
        if kind == "teammate":
            result["teammate_src"] = remap[graph["teammate_src"][keep]]
    return result


def pack(graphs: list[Graph]) -> Graph:
    """Concatenate team graphs with shifted agent indices and one env index per team."""
    pieces: dict[str, list[np.ndarray]] = {k: [] for k in graphs[0]}
    offset = 0
    for env, graph in enumerate(graphs):
        for key, value in graph.items():
            if key == "agent_env":
                value = np.full(len(value), env, np.int64)
            elif key in INDICES:
                value = value + offset
            pieces[key].append(value)
        offset += len(graph["self"])
    return {k: np.concatenate(values) for k, values in pieces.items()}


@dataclass
class Transition:
    observation: Graph
    actions: np.ndarray
    reward: float
    next_observation: Graph
    discount: float


class NStep:
    """Flush at termination or truncation; only true termination removes the bootstrap."""

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.pending: deque = deque()

    def append(
        self,
        obs: Graph,
        actions: np.ndarray,
        reward: float,
        next_obs: Graph,
        terminated: bool,
        truncated: bool,
    ) -> list[Transition]:
        self.pending.append((obs, actions.copy(), reward, next_obs, terminated))
        out = []
        while len(self.pending) >= self.n or ((terminated or truncated) and self.pending):
            steps = list(self.pending)[: self.n]
            total = sum(self.gamma**i * step[2] for i, step in enumerate(steps))
            out.append(
                Transition(
                    steps[0][0],
                    steps[0][1],
                    total,
                    steps[-1][3],
                    0.0 if steps[-1][4] else self.gamma ** len(steps),
                )
            )
            self.pending.popleft()
        return out


class Replay:
    """Uniform sampling from a bounded ring of complete team transitions."""

    def __init__(self, capacity: int, rng: np.random.Generator, compress: bool = True):
        self.capacity, self.rng = capacity, rng
        self.compress = compress
        self.items: list[Transition | bytes] = []
        self.next = 0

    def add(self, transition: Transition) -> None:
        # Repeated small graph arrays otherwise spend more RAM on Python objects than
        # features. Only self-produced objects are decoded; no external pickle is read.
        stored = (
            zlib.compress(pickle.dumps(transition, protocol=5), level=1)
            if self.compress
            else transition
        )
        if len(self.items) < self.capacity:
            self.items.append(stored)
        else:
            self.items[self.next] = stored
        self.next = (self.next + 1) % self.capacity

    def sample(self, count: int) -> list[Transition]:
        selected = [self.items[i] for i in self.rng.integers(len(self.items), size=count)]
        return [pickle.loads(zlib.decompress(x)) if isinstance(x, bytes) else x for x in selected]

    def __len__(self) -> int:
        return len(self.items)
