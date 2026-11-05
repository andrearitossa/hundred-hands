"""Local-first experiment records with W&B offline/online logging from the same metrics."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import wandb
import yaml


class Run:
    """Record resolved config, seed, revision and metrics; register checkpoints as artifacts."""

    def __init__(
        self,
        path: Path,
        config: dict,
        mode: Literal["online", "offline", "disabled", "shared"] = "offline",
        job: str = "train",
    ):
        self.path = path
        self.started = time.perf_counter()
        path.mkdir(parents=True, exist_ok=True)
        config = dict(config)
        if isinstance(config.get("env"), str):
            env_path = Path(config["env"])
            resolved = yaml.safe_load(env_path.read_text())
            resolved["limits"] = yaml.safe_load((env_path.parent / resolved["limits"]).read_text())
            if config.get("red_count"):
                resolved["red"]["count_min"] = resolved["red"]["count_max"] = config["red_count"]
            if config.get("curriculum"):
                resolved["red"]["behaviour_weights"] = dict(
                    zip(
                        ("straight", "waypoint", "evasive"),
                        config["curriculum"][0]["weights"],
                        strict=True,
                    )
                )
            config["resolved_env"] = resolved
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        for key, folder in (
            ("WANDB_CACHE_DIR", ".cache"),
            ("WANDB_CONFIG_DIR", ".config"),
            ("WANDB_DATA_DIR", ".data"),
        ):
            os.environ.setdefault(key, str((path / folder).resolve()))
        self.run = wandb.init(
            project="hundred-hands-v0",
            mode=mode,
            dir=str(path),
            config=config,
            job_type=job,
            tags=[sha],
        )
        sources = {}
        for package in (
            "core",
            "sim",
            "perception",
            "gnc",
            "policy",
            "bindings",
            "eval",
            "scenarios",
            "tools",
        ):
            for source in Path(package).rglob("*"):
                if source.is_file() and (
                    source.suffix in (".py", ".cpp", ".hpp", ".yaml")
                    or source.name == "CMakeLists.txt"
                ):
                    sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        (path / "source-manifest.json").write_text(json.dumps(sources, indent=2))
        self.sha = sha
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "git_sha": sha,
                    "started_at": datetime.now(UTC).isoformat(),
                    "command": sys.argv,
                    "cpu_count": os.cpu_count(),
                    "cpu": next(
                        (
                            line.split(":", 1)[1].strip()
                            for line in Path("/proc/cpuinfo").read_text().splitlines()
                            if line.startswith("model name")
                        ),
                        platform.processor(),
                    ),
                    "platform": platform.platform(),
                    "wandb_url": self.run.url,
                    "wandb_mode": mode,
                    "seed": config.get("seed"),
                    "working_tree_dirty": bool(
                        subprocess.check_output(["git", "status", "--porcelain"], text=True)
                    ),
                },
                indent=2,
            )
        )

    def log(self, metrics: dict, step: int | None = None) -> None:
        self.run.log(metrics, step=step)
        with (self.path / "history.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    def checkpoint(self, path: Path) -> None:
        artifact = wandb.Artifact(
            f"gqn-{self.run.id}",
            type="model",
            metadata={"config": dict(self.run.config), "git_sha": self.sha},
        )
        artifact.add_file(str(path))
        self.run.log_artifact(
            artifact, aliases=["best", "latest"] if path.name == "best.pt" else ["latest"]
        )

    def finish(self) -> None:
        self.run.finish()
        path = self.path / "meta.json"
        meta = json.loads(path.read_text())
        meta.update(
            duration_s=time.perf_counter() - self.started, finished_at=datetime.now(UTC).isoformat()
        )
        path.write_text(json.dumps(meta, indent=2))


def resolve_checkpoint(reference: str, cache: Path) -> Path:
    """Resolve a local checkpoint/directory, W&B run URL, or model artifact reference."""
    from urllib.parse import urlparse

    local = Path(reference)
    if local.exists():
        if local.is_file():
            return local
        candidates = [local / "best.pt", local / "last.pt", *sorted(local.glob("*.pt"))]
        found = next((p for p in candidates if p.is_file()), None)
        if found:
            return found
        raise FileNotFoundError(f"No checkpoint in {local}")
    if reference.startswith("wandb://"):
        identifier = reference.removeprefix("wandb://")
    elif reference.startswith("https://"):
        parsed = urlparse(reference)
        if parsed.hostname not in ("wandb.ai", "www.wandb.ai"):
            raise ValueError("Expected a wandb.ai run URL")
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 4 or parts[2] != "runs":
            raise ValueError("Expected /entity/project/runs/id")
        identifier = "/".join((parts[0], parts[1], parts[3]))
    else:
        identifier = reference
    api = wandb.Api()
    if ":" in identifier:
        artifact = api.artifact(identifier, type="model")
    else:
        run = api.run(identifier)
        artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
        if not artifacts:
            raise FileNotFoundError("W&B run has no logged model artifact")
        preferred = [a for a in artifacts if "best" in a.aliases]
        artifact = max(preferred or artifacts, key=lambda a: int(a.version.removeprefix("v")))
    directory = Path(artifact.download(root=str(cache)))
    return resolve_checkpoint(str(directory), cache)
