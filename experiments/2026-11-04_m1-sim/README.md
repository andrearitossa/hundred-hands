# M1: simulation core

**Question:** does the world obey the spec's rules, and do idle defenders always lose?

**Outcome:** yes, on what was measured. Asset hit in 1000/1000 idle episodes for each of `straight`, `waypoint` and `evasive`, with no truncations.
13 C++ tests and the 3 pytest invariant cases (200 seeds each) pass.

> Follow-up: the single-attacker, zero-defender check found failures hidden by this
> multi-attacker test. See [the no-defence iteration](../2026-11-04_no-defence/README.md)
> for the fix and stronger checks.

## Results (`sim.json`, 1000 episodes per behaviour, same seeds)

| Behaviour | Asset hit | Median episode | Red building crashes | Jinks | Decisions/s (1 thread, via Python) |
| --- | --- | --- | --- | --- | --- |
| straight | 100 % | 48.7 s | 171 of 4526 spawned | 0 | 5165 |
| waypoint | 100 % | 60.5 s | 0 | 0 | 5306 |
| evasive | 100 % | 53.7 s | 97 | 605 | see `sim.json` |

Crash counts are lower bounds: each episode stops at the first asset hit, so later attackers never finish flying.

## Look at it

- `replay_*.html`: top-down replay in a browser. Dashed lines are planned routes, X marks losses and intercepts.
- `crashes_*.png`: zoomed crash geometry. `crashes_*.jsonl`: one line per crash.

## Build journal

1. **First run looked perfect, and was wrong.** 100 % hits but zero building crashes out of 886 attackers.
   Rule tests confirmed contacts never fired: MuJoCo also filters on `body_contype`, which is fixed at compile, and
   drones compile with collisions off. Fix: `Env::set_collidable` sets both the geom and body masks.
2. **Waypoint attackers crashed on approved routes** (19 crashes, 0–2 m from the route). An independent slab test
   showed `mj_ray` itself was right; it only tests a zero-width line.
3. **A 30 m corridor overcorrected.** Routes near spawn became infeasible and fell back to a straight line (`empty=26`).
4. **The crash gallery showed both causes:** rays passing just over roofs (z=32.9 vs h≈33), and buildings about 60 m
   ahead of a spawn. Fix: rays sweep below the body, corridor 15 m, lateral scatter 200 m. Result: 0 crashes, 0 empty routes.

## Assumptions and deviations (please check)

- `velocity_control` lives in `core/`, not `gnc/`. Attackers in `sim` need it, and `sim` may not depend on `gnc`.
- `spawn_clearance: 60` m interprets the spec's undefined "spawn ring". Added during step 3 and not re-tested since the corridor narrowed.
- Waypoint altitude stays at the attacker's cruise altitude; the dive is included in the route check.
- `straight` and `evasive` attackers can still hit buildings, as the spec's rules allow. No episode ended without an asset hit here.
  It remains possible if every attacker in an episode crashes, so the 100 % invariant is measured, not guaranteed.
- `v_max` and `vz_max` are assumed, not derived (`scenarios/assets/x500/README.md`).
- The world is recompiled at every reset, for simplicity. Not yet measured as a share of runtime.

## Not verified

- Replay HTML: only its JavaScript syntax was checked. It has not been opened in a browser.
- Enforcement of the vendored limits at rigid-body level: out of scope for point mass.
- MuJoCo viewer replay: not built; only the HTML fallback exists.
- M0 gaps still open: CI, dependency-rule check, codegen, `CLAUDE.md`, `README`, macOS.
