"""Dynamic packed-graph ONNX export and logged-observation parity checking."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import INPUTS, tensors
from .train import load

OUTPUTS = ("q", "attention_local", "attention_global", "attention_teammate", "messages")


def export(model, graph: dict, path: Path) -> None:
    """Export a CPU GQN with independent dynamic counts for agents and each relation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    axes = {
        name: {0: group}
        for name, group in zip(
            INPUTS,
            (
                "agents",
                "agents",
                "local_nodes",
                "local_nodes",
                "global_nodes",
                "global_nodes",
                "teammate_nodes",
                "teammate_nodes",
                "teammate_nodes",
                "agents",
            ),
            strict=True,
        )
    }
    axes.update(
        {
            name: {0: group}
            for name, group in zip(
                OUTPUTS,
                ("agents", "local_nodes", "global_nodes", "teammate_nodes", "agents"),
                strict=True,
            )
        }
    )
    torch.onnx.export(
        model.cpu().eval(),
        tensors(graph),
        str(path),
        input_names=list(INPUTS),
        output_names=list(OUTPUTS),
        dynamic_axes=axes,
        opset_version=18,
        dynamo=False,
    )


def check(model, path: Path, graphs: list[dict]) -> dict:
    """Compare all Q-values through Python ORT and, when built, the C++ runtime."""
    import onnxruntime as ort
    import swarmsim

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    runtime = swarmsim.Runtime(str(path))
    worst = 0.0
    for graph in graphs:
        inputs = tensors(graph)
        with torch.no_grad():
            expected = model(*inputs)[0].numpy()
        actual = session.run(
            None, {key: value.numpy() for key, value in zip(INPUTS, inputs, strict=True)}
        )[0]
        cpp = np.asarray(runtime.forward(graph)["q"])
        for output in (actual, cpp):
            if expected.size:
                worst = max(worst, float(np.max(np.abs(output - expected))))
    if worst >= 1e-4:
        raise AssertionError(f"ONNX Q parity error {worst}")
    return {"observations": len(graphs), "max_absolute_error": worst}


def main():
    import swarmsim

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--observations", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load(args.checkpoint)
    env = swarmsim.TeamEnv(swarmsim.EnvConfig.load("scenarios/v0/env.yaml"))
    graph = env.reset(20000000)
    path = args.out or args.checkpoint.with_suffix(".onnx")
    export(model, graph, path)
    graphs = []
    seed = 20000000
    for _ in range(args.observations):
        graphs.append(graph)
        with torch.no_grad():
            actions = model(*tensors(graph))[0].argmax(-1).tolist()
        result = env.step(actions)
        graph = result["observation"]
        if result["terminated"] or result["truncated"]:
            seed += 1
            graph = env.reset(seed)
    metrics = check(model, path, graphs)
    np.savez_compressed(
        path.with_suffix(".observations.npz"),
        graphs=np.array([json.dumps({k: v.tolist() for k, v in g.items()}) for g in graphs]),
    )
    path.with_suffix(".parity.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
