"""Native ONNX inference must preserve dynamic graph behaviour, including empty teams."""

from pathlib import Path

import numpy as np
import swarmsim
import torch
from swarm_policy.export import check, export
from swarm_policy.model import GQN
from swarm_policy.replay import split


def test_cpp_export_dynamic_shapes(tmp_path: Path):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    env = swarmsim.TeamEnv(swarmsim.EnvConfig.load("scenarios/v0/env.yaml"))
    initial = env.reset(2)
    graph = initial
    graphs = [initial, split(initial, 999)]
    for _ in range(100):
        graph = env.step([13] * len(graph["self"]))["observation"]
    graphs.append(graph)
    model = GQN()
    path = tmp_path / "model.onnx"
    export(model, initial, path)
    assert check(model, path, graphs)["max_absolute_error"] < 1e-4
    runtime = swarmsim.Runtime(str(path))
    a = runtime.forward(graph)["q"]
    assert np.isfinite(a).all()
