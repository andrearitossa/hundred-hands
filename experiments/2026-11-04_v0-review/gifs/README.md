# Environment GIFs

These are real `swarmsim.TeamEnv` episodes rendered as 3D perspective views with
matplotlib and written with Pillow. The axes are world x/y/z in metres; buildings
are 3D boxes, the asset is the green marker on the ground, defenders are blue
(`B0`–`B2`), and attackers are red. Faint lines are full 3D flight trails; vertical
lines show each aircraft's height above the ground.

| GIF | Seed | Policy | Outcome |
| --- | ---: | --- | --- |
| `idle_asset_hit.gif` | 23 | idle | 3 straight attackers hit the asset at 26.8 s; all defenders survive because they never move |
| `pn_greedy_defence.gif` | 24 | PN + greedy | asset intact at 39.6 s; all 3 defenders survive |
| `gqn_current_policy.gif` | 10,000,000 | current M7 GQN checkpoint | asset hit at 52.2 s; all defenders left the area / were lost |

All three use exactly three attackers with `straight` behaviour and the current
`scenarios/v0/env.yaml`. The GQN clip is a diagnostic of the current learner, not
a success claim; the M7 checkpoint was trained for only 8,000 decisions.

Regenerate with:

```bash
MPLCONFIGDIR=/tmp/hundred-hands-mpl \
  .venv/bin/python experiments/2026-11-04_v0-review/make_gifs.py
```
