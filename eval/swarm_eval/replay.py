"""Self-contained top-down HTML replay of a recorded episode (or several)."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import swarmsim

TEMPLATE = (Path(__file__).parent / "replay_template.html").read_text()


def _to_json(ep: dict) -> dict:
    step = max(1, len(ep["t"]) // 1500)
    routes = ep.get("routes")
    if routes is None and "route_offsets" in ep:
        routes = [
            ep["route_points"][a:b]
            for a, b in zip(ep["route_offsets"][:-1], ep["route_offsets"][1:], strict=True)
        ]
    raw_decisions = np.atleast_1d(ep.get("decisions", []))
    decisions = [
        json.loads(str(x))
        for x in raw_decisions[::step]
        if isinstance(x, (str, np.str_)) and str(x).strip()
    ]
    decisions = [x for x in decisions if isinstance(x, dict) and "graph" in x and "decision" in x]
    return {
        "decisions": decisions,
        "seed": int(ep["seed"]),
        "nBlue": int(ep["n_blue"]),
        "t": np.round(ep["t"][::step], 2).tolist(),
        "pos": np.round(ep["states"][::step, :, :3], 1).tolist(),
        "status": ep["status"][::step].astype(int).tolist(),
        "asset": np.round(ep["asset"], 1).tolist(),
        "buildings": np.round(ep["buildings"], 1).tolist(),
        "routes": [np.round(r, 1).tolist() for r in routes or []],
        "behaviour": [swarmsim.BEHAVIOURS[int(b)] for b in ep["red"][:, 2]],
        "events": [
            {
                "t": round(float(e[0]), 2),
                "kind": swarmsim.EVENT_KINDS[int(e[1])],
                "drone": int(e[2]),
                "pos": np.round(
                    ep["states"][int(np.abs(ep["t"] - e[0]).argmin()), max(int(e[2]), 0), :2], 1
                ).tolist(),
                "cause": swarmsim.LOSS_CAUSES[int(e[4])],
            }
            for e in ep["events"]
        ],
        "assetHit": bool(ep["asset_hit"]),
        "intercepted": int(ep["intercepted"]),
    }


def write_html(episodes: list[dict], path: Path, title: str = "Episode replay") -> None:
    """Write `episodes` to one standalone HTML file with an episode picker."""
    data = json.dumps([_to_json(e) for e in episodes], separators=(",", ":"))
    path.write_text(TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episodes", nargs="+", type=Path, help=".npz episode logs")
    ap.add_argument(
        "--viewer", action="store_true", help="Open the recorded scene in the MuJoCo viewer"
    )
    ap.add_argument("-o", "--out", type=Path, default=Path("replay.html"))
    a = ap.parse_args()
    episodes = [dict(np.load(p)) for p in a.episodes]
    if a.viewer:
        viewer(episodes[0])
    write_html(episodes, a.out)
    print(a.out)


def viewer(ep: dict) -> None:
    """Replay point-mass states and global attention lines in MuJoCo's interactive viewer."""
    import time
    from xml.etree.ElementTree import Element, SubElement, tostring

    import mujoco
    import mujoco.viewer

    root = Element("mujoco")
    world = SubElement(root, "worldbody")
    SubElement(world, "geom", type="plane", size="500 500 0.1", rgba="0.8 0.8 0.8 1")
    for b in ep["buildings"]:
        SubElement(
            world, "geom", type="box", pos=" ".join(map(str, b[:3])), size=" ".join(map(str, b[3:]))
        )
    SubElement(
        world,
        "geom",
        type="sphere",
        pos=" ".join(map(str, ep["asset"])),
        size="10",
        rgba="0 0.6 0 1",
    )
    n = ep["states"].shape[1]
    for i in range(n):
        body = SubElement(world, "body", name=f"drone{i}")
        SubElement(body, "freejoint")
        SubElement(
            body,
            "geom",
            type="sphere",
            size="0.4",
            mass="1",
            rgba="0 0 1 1" if i < int(ep["n_blue"]) else "1 0 0 1",
        )
    model = mujoco.MjModel.from_xml_string(tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as window:
        window.cam.distance = 800
        for frame, t in enumerate(ep["t"]):
            if not window.is_running():
                break
            for i in range(n):
                data.qpos[7 * i : 7 * i + 3] = ep["states"][frame, i, :3]
                data.qpos[7 * i + 3 : 7 * i + 7] = [1, 0, 0, 0]
                model.geom_rgba[1 + len(ep["buildings"]) + 1 + i, 3] = float(
                    ep["status"][frame, i] == 1
                )
            mujoco.mj_forward(model, data)
            window.user_scn.ngeom = 0
            logs = ep.get("decisions", [])
            if frame < len(logs):
                log = json.loads(logs[frame])
                graph = log["graph"]
                for node, owner, weights in zip(
                    graph["global_threat"],
                    graph["global_owner"],
                    log["decision"]["attention_global"],
                    strict=True,
                ):
                    drone = graph["agent_drone"][owner]
                    start = ep["states"][frame, drone, :3]
                    psi = log["headings"][owner]
                    x, y, z = np.array(node[:3]) * 1000
                    end = start + np.array(
                        [x * np.cos(psi) - y * np.sin(psi), x * np.sin(psi) + y * np.cos(psi), z]
                    )
                    scene = window.user_scn
                    if scene.ngeom >= scene.maxgeom:
                        break
                    geom = scene.geoms[scene.ngeom]
                    scene.ngeom += 1
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3),
                        np.zeros(3),
                        np.eye(3).ravel(),
                        np.array([1, 0.5, 0, float(np.mean(weights))], dtype=np.float32),
                    )
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2, start, end)
            window.sync()
            if frame + 1 < len(ep["t"]):
                time.sleep(float(ep["t"][frame + 1] - t) / 20)


if __name__ == "__main__":
    main()
