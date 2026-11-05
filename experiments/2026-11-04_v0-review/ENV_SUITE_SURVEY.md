# Online survey: existing environments for “this kind of task”

Checked online on 2026-11-04 via GitHub and project documentation.
“This kind of task” here means: multi-agent, cooperative defence of a fixed asset
against moving attackers, with 3D or 2D motion, partial/limited sensing, variable
numbers of agents, and a policy over graph/relational observations.

## Short answer

**No widely used, maintained benchmark exists for this exact task.** There is no
“SMAC/MPE for drone asset defence” that matches 3D point-mass drones, scripted
attackers, range-limited perception, fused tracks, variable team size and a GQN
policy.

The closest items are:

1. **MARSim** — a small, research-prototype multi-agent UAV/UGV resupply and
   interception environment. Conceptually the closest, but grid-based and not an
   established benchmark.
2. **MPE2 / PettingZoo `simple_tag` and `simple_world_comm`** — 2D predator-prey
   and communication environments; closest *benchmark semantics* for adversarial
   multi-agent pursuit.
3. **gym-pybullet-drones**, **PyFlyt**, **Aerial Gym Simulator** — drone/quadrotor
   simulation and control, not asset-defence tasks.
4. **SMAC / SMACv2** — strong multi-agent cooperative benchmark, but StarCraft
   units, not drones or graph observations.

## Closest domain/task matches

| Project | What it is | Fit | Main gap |
| --- | --- | --- | --- |
| [MARSim](https://github.com/McFadden-S/MARSim) | Gymnasium-compatible grid-based multi-agent resupply simulator; friendly UAV/UGV vs enemy UAV/UGV; PPO baseline; procedural maps; collision/kamikaze modes | Direct conceptual match: contested logistics, UAV interception, adversarial teams | Grid abstraction, small/new project (~4 stars at check), no 3D dynamics, no perception/fusion or graph observation, not a benchmark suite |
| [MPE2](https://github.com/Farama-Foundation/MPE2) | Maintained Multi Particle Environments; `simple_tag`, `simple_world_comm`, `simple_spread`, etc. | Standard 2D multi-agent pursuit/communication testbed; easy external sanity check for MARL algorithms | 2D particles, small fixed teams, no drone dynamics, no asset-defence benchmark |
| [MAgent2](https://github.com/Farama-Foundation/MAgent2) | Large multi-agent grid battles and competitive scenarios, PettingZoo API | Large-team predator/prey and adversarial pressure | Grid/war-game abstraction, not 3D drone control or perception |
| [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | Standard multi-agent API and environment collection | Best API to expose a custom defence env to MARL tooling | API/framework, not a task benchmark |

## Drone/quadrotor environments

| Project | What it is | Fit | Main gap |
| --- | --- | --- | --- |
| [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) | PyBullet/Gymnasium single- and multi-agent quadcopter control; multi-agent aviaries/swarms; stable-baselines3 examples | Best-known lightweight drone RL environment; good dynamics/control reference | No scripted attackers, asset-defence task, fused perception or GQN benchmark |
| [PyFlyt](https://github.com/jjshoots/PyFlyt) | UAV RL simulator with Gymnasium and PettingZoo environments; quadrotor hover/waypoints/gates, multi-agent hover, fixed-wing dogfight | Closest 3D pursuit/evasion and multi-agent UAV control | Dogfight is fixed-wing; no asset-defence/limited-sensing benchmark |
| [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) | Isaac Gym-based, highly parallel MAV simulator with thousands of multirotors, controllers, LiDAR/camera | Scale and high-fidelity physics for future larger swarms | GPU/Isaac Gym dependency; navigation/control focus; no adversarial asset-defence task |
| [safe-control-gym](https://github.com/utiasDSL/safe-control-gym) | Safe RL benchmark for cart-pole/quadrotor with symbolic constraints | Safety/constraint baseline for control | Single-agent focus; not multi-agent defence |
| [crazyflow](https://github.com/learnsyslab/crazyflow) | JAX-based, GPU-parallel quadrotor simulation | Fast differentiable/parallel drone experiments | Quadrotor control, not adversarial MARL |

Other simulators worth knowing but not directly an RL env suite:
**AirSim / Project AirSim**, **Flightmare**, **Pegasus Simulator**, **Gazebo/ROS 2**.
They provide realistic simulation but not the task definition, benchmark seeds,
metrics or baselines for this problem.

## Cooperative MARL benchmark suites

These are useful for validating the learner and reporting conventions, even though
the domain is different:

| Project | Scope | Why it matters here |
| --- | --- | --- |
| [SMAC](https://github.com/oxwhirl/smac) / [SMACv2](https://github.com/oxwhirl/smacv2) | StarCraft II decentralised micromanagement; cooperative MARL benchmark | Standard comparison point for VDN/QMIX-style methods and evaluation protocol |
| [JaxMARL](https://github.com/FLAIROx/JaxMARL) | JAX implementations and environments including SMAX, MPE, Overcooked, etc. | Fast MARL experimentation; useful if the learner is ever ported to JAX |
| [BenchMARL](https://github.com/facebookresearch/BenchMARL) | Benchmarking framework for MARL algorithms/environments | Could structure external comparisons, not a task environment |
| [MARLlib](https://github.com/Replicable-MARL/MARLlib) | Large MARL algorithm/environment integration library | Reference implementations and environment adapters |
| [Melting Pot](https://github.com/google-deepmind/meltingpot) | Multi-agent generalization/substrate suite | Generalization and social/cooperative evaluation, not drone defence |
| [Neural MMO](https://github.com/NeuralMMO/environment) | Large-scale multi-agent survival | Scale/generalization inspiration, far from drone dynamics |

## What this means for Hundred Hands

The custom environment is justified. The exact combination is not covered by an
existing suite:

- 3D point-mass drones with X500-derived limits.
- Scripted red behaviours and a fixed asset.
- Range-limited clean sensing, per-defender trackers, fusion and no ground-truth
  policy input.
- Variable attacker/defender counts and ego-centric relational graphs.
- A GQN/VDN learner plus PN/PP assignment baselines.
- C++ env/perception/control and Python learner split.

Recommendations:

1. **Do not replace the custom env.** Borrow conventions, not the task.
2. Use **MPE2 `simple_tag`** as a cheap external sanity check for the MARL code
   path: can the same GQN/VDN machinery learn a small cooperative pursuit task?
3. Use **SMAC/SMACv2** only as a protocol/reference comparison if the paper needs
   external MARL context; it is not a drone environment.
4. If higher-fidelity drone dynamics are ever required, consider
   **gym-pybullet-drones** or **PyFlyt** as a replacement physics layer, keeping the
   defence logic/perception above it.
5. If a publication needs “closest prior environments”, cite **MARSim** for the
   contested UAV/UGV task concept, **MPE2 `simple_tag`** for predator-prey MARL, and
   **gym-pybullet-drones/PyFlyt/Aerial Gym** for drone RL. Explain that none
   combines 3D dynamics, limited perception, variable graph observations and
   asset defence.

## Caveats

- Star counts and repository activity are snapshots from the check date.
- MARSim is very new/small; inspect its code and licence before relying on it.
- “No exact benchmark” is a conclusion from the checked sources, not a formal
  literature review. A paper-level survey could still find niche research
  environments that are not packaged as maintained suites.


## Code reuse and licensing

See [REUSE_ANALYSIS.md](REUSE_ANALYSIS.md) for the licence matrix and a concrete
reuse plan. Summary: all major candidates are MIT/BSD-3/Apache-2.0 and can be
reused inside this Apache-2.0 repo with notices, but the Python/PyBullet/Isaac Gym
backends do not fit the current C++/MuJoCo point-mass design. Borrow interfaces,
task ideas and evaluation conventions; do not copy whole environment loops.
