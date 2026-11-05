"""Generate an inspectable experiment table and plot from evaluator metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def report(path: Path) -> None:
    """Write report.md and plots/survival.png; metrics remain the single source of truth."""
    metrics = json.loads((path / "metrics.json").read_text())
    rows = metrics.get("evaluation", [])
    text = [
        "# Evaluation report",
        "",
        "| Policy | Red | Behaviour | Survival (95% CI) | Intercept fraction | Truncated |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        interval = row["survival_ci95"]
        text.append(
            f"| {row['policy']} | {row['red']} | {row['behaviour']} | {row['survival']:.3f} [{interval[0]:.3f}, {interval[1]:.3f}] | {row['intercept_fraction']:.3f} | {row['truncated']} |"
        )
    if rows:
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.4), 4))
        ax.bar(range(len(rows)), [r["survival"] for r in rows])
        ax.set_xticks(
            range(len(rows)),
            [f"{r['policy']}\n{r['red']} {r['behaviour']}" for r in rows],
            rotation=90,
        )
        ax.set(ylabel="Asset survival", ylim=(0, 1))
        fig.tight_layout()
        (path / "plots").mkdir(exist_ok=True)
        fig.savefig(path / "plots" / "survival.png", dpi=130)
        plt.close(fig)
        text.extend(["", "![Asset survival](plots/survival.png)"])
    else:
        text.extend(["", "```json", json.dumps(metrics, indent=2), "```"])
    update_readme()
    (path / "report.md").write_text("\n".join(text) + "\n")


def update_readme() -> None:
    """Regenerate only the marked README results block from recorded evaluation metrics."""
    root = Path(__file__).resolve().parents[2]
    readme = root / "README.md"
    if not readme.exists():
        return
    text = readme.read_text()
    if "<!-- results:start -->" not in text:
        return
    lines = [
        "| Experiment | Policy | Red / behaviour | Episodes | Asset survival |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for metrics in sorted((root / "experiments").glob("*/metrics.json")):
        data = json.loads(metrics.read_text())
        for row in data.get("evaluation", []):
            relative = metrics.parent.relative_to(root)
            lines.append(
                f"| [{metrics.parent.name}]({relative}/report.md) | {row['policy']} | {row['red']} / {row['behaviour']} | {row['episodes']} | {row['survival']:.1%} |"
            )
    before = text.split("<!-- results:start -->")[0]
    after = text.split("<!-- results:end -->")[1]
    readme.write_text(
        before + "<!-- results:start -->\n" + "\n".join(lines) + "\n<!-- results:end -->" + after
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    report(parser.parse_args().path)


if __name__ == "__main__":
    main()
