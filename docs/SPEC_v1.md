# SPEC — Hundred Hands, v1: the paper stack

Status: draft for review · Builds on: [`SPEC_v0.md`](SPEC_v0.md) · Audience: engineers and Claude Code

**Entry gate:** do not start v1 until the v0 end-to-end skeleton is working and documented (v0 M7, §13.1).

v1 is the **minimal, fully functional stack** for one publication, built around one thesis:

> **Training with dropout changes how a swarm coordinates, and that change explains how it copes with
> losing members, or it doesn't.**

Either answer is a result. Anything that does not help test this thesis belongs in
[`SPEC_v2.md`](SPEC_v2.md).

Working title: *What Holds a Swarm Together? Interpreting Coordination and Its Breakdown in
Decentralised Graph Q-Network Swarm Defence.*

---

## 1. Goal

v1 is done when:

1. **Thesis:** RQ5 (§2) has an answer backed by interventions on both GQN policies, or a recorded negative result.
2. **Mechanism:** RQ1–RQ4 each have an answer backed by an intervention, or a recorded negative result.
3. **Performance context:** both GQN policies have been compared with PN + greedy on asset survival, across training seeds, with and without dropout.
4. **Reproducibility:** every figure in the paper regenerates from `experiments/`, using frozen policies (§7).

The comparison has no required winner. A loss to PN + greedy, explained, is still a result.

---

## 2. Research questions

RQ5 is the thesis. RQ1–RQ2 supply the measurements of mechanism, and RQ3–RQ4 supply the conditions they are measured under.

| # | Question | Observation | Causal test |
| --- | --- | --- | --- |
| RQ1 | What does each drone represent? Does attention recover an implicit assignment? | Attention vs Hungarian agreement; linear probes for time-to-intercept and threat-to-asset distance | Ablate or steer along the probed direction; check behaviour changes as predicted |
| RQ2 | What do messages carry? Do they encode the sender's intended target? | Decode the sender's next attended threat from `m_i` | Zero, swap and replay messages; measure change in `Q`, action and outcome |
| RQ3 | When k of n drones are **destroyed**, how fast and how well do survivors reallocate, and through which internal change? | Time to re-cover threats; attention shift; Q-margin | Paired branches with and without the loss |
| RQ4 | When drones are **disconnected** but alive, do they act sensibly alone or duplicate the swarm's effort? And on **reconnecting**, how fast do they re-coordinate? | Duplicate engagement; coverage; re-coordination time | Paired branches; `g = 0` against the same scene with `g = 1` |
| RQ5 | Does training with dropout change the **mechanism**, not only the score, and does that change explain its behaviour under dropout? | RQ1–RQ4 metrics, compared between policies | The same interventions on both policies |

**Evidence rule:** attention weights alone never support a claim. Every mechanism claim needs an intervention.

---

## 3. Dropout model

| Event | Meaning | Reversible |
| --- | --- | --- |
| **Destroy** | The drone is removed from the world | No |
| **Disconnect** | Alive and flying with `g_i = 0`. It neither sends nor receives: it acts on its local branch and own tracker, and its node and tracks leave teammates' graphs | Yes |
| **Reconnect** | `g_i = 1` again. Its tracks and message return on the same decision step | — |

- API: `schedule(event, drones, t)` in config or by call, varying k, timing and selection rule (random, nearest to a threat, currently engaged).
- Baselines use the same rules: a disconnected baseline uses only its own tracker and stops being assigned by the team.
- `Q_team` sums over drones present in the world, including disconnected ones. Destroy changes the set.
- Training mode (`dropout.train`): random disconnect, reconnect and destroy, with configured rates. Off for the no-dropout policy.
- Comms stay otherwise ideal. Range, loss and delay are v2 modules.

---

## 4. Paired counterfactuals

Same episode, with and without the intervention. This replaces v0's "determinism not required".

- `Env::snapshot()` / `restore()` capture all env state: MuJoCo state, statuses, red scripts, trackers, decoders, RNG.
- Branch at t*, apply the intervention to one branch only.
- **Common random numbers:** an intercept trial's draw derives from (episode seed, blue, red, entry count), so a branch changes outcomes only through behaviour.
- Scope: reproducible on one machine and build, not across machines.
- Acceptance: a branch with no intervention matches the original **exactly** over a full episode.
- Policies are evaluated greedily, so action selection adds no randomness.

---

## 5. Policies compared

| Policy | Purpose |
| --- | --- |
| GQN, no dropout training | Reference |
| GQN, dropout training | The thesis (RQ5) |
| PN + greedy; PP/PN × greedy/Hungarian | Performance context |
| Random, idle | Sanity floors |

- **Training seeds:** 5 per learned policy (3 is the minimum if compute forces it; say so in the paper).
- Full v0 curriculum and behaviour mix. Training runs on Colab or cloud; budget is set from measured throughput.
- Final network size is chosen here using the v0 latency rule.

---

## 6. Analysis toolkit (`eval/`)

Run on any policy checkpoint. Each tool writes metrics and plots to its experiment folder.

| Tool | Does |
| --- | --- |
| Assignment | Implied assignment vs Hungarian over time; coverage gaps; duplicate engagement |
| Probes | Linear probes on embeddings, with held-out seeds and a shuffled-label control |
| Message interventions | Zero, swap between teammates, replay from an earlier time |
| Branch ablations | Local only (`g = 0`), messages zeroed, attention block ablated at t* |
| Dropout sweep | Survival and recovery against k, timing and selection rule, paired |
| Episode viewer | Replay with attention, dropout events, and paired branches side by side |

One retrained ablation, run only if a finding depends on relation-typed attention: single attention with a one-hot kind, as its own experiment.

---

## 7. Improvement loop

This is how the policies get good enough to study. It is not a claim of the paper or a v1 done criterion.

Each iteration is a journal entry: **finding → hypothesis → change → re-measure → keep or drop**.

- The hypothesis and metric are written before the retrain.
- Results use held-out eval seeds and the multi-seed protocol.
- Candidate changes come from findings. They may include shared team-level shaping (v0 §7.3 fallback) and dropout schedules.
- Failed attempts stay in the record.
- A kept change that is not itself dropout training applies to both GQN policies, so they remain comparable.
- **Freeze:** the loop ends before final experiments. Every paper result comes from the frozen policies.

---

## 8. Statistics and reporting

- Report per-seed results and aggregate across seeds. Training seed is the unit of replication, not the episode.
- Paired episodes: paired tests (McNemar for survival, bootstrap for continuous metrics).
- 95 % CIs from a hierarchical bootstrap over seeds, then episodes. Correct for multiple comparisons across RQs.
- A finding is reported as consistent only with the count of seeds showing it.
- **Scope statement in the paper:** point-mass dynamics, clean sensing, ideal comms apart from dropout. Whether results survive realism is left to v2 modules, unless one is run and reported.
- Check citations before use. Starting points from scoping, not yet verified: *Probing Dec-POMDP Reasoning in Cooperative MARL*; *Interpretable Failure Analysis in MARL Systems*; *Interpreting Emergent Planning in Model-Free RL*; Bouton et al. (arXiv 2302.01199); *GNNs for Decentralized Multi-Agent Perimeter Defense* (arXiv 2301.09689).

---

## 9. Tests (added)

| Area | Test |
| --- | --- |
| Snapshot | No-intervention branch is identical; restore is idempotent; trackers and decoders restored |
| Common random numbers | The same trial gets the same draw across branches |
| Dropout | Disconnected drones are absent from teammates' graphs and fused picture, and see `g = 0`; reconnect restores them on the same step; destroy is permanent |
| Dropout training | Observed event rates match config; replay handles changing team sets |
| Probes | Recover a planted signal; shuffled labels give chance level |
| Interventions | Zeroing and swapping change only the intended inputs |

---

## 10. Milestones (continuing from v0 M7)

| # | Milestone | Acceptance check |
| --- | --- | --- |
| M8 | Snapshot and branch | Identical no-intervention branch; common random numbers |
| M9 | Dropout model | Three events through the API and viewer; baselines follow the same rules |
| M10 | **Performance baseline** | No-dropout GQN, multi-seed, full mix; suite vs baselines, and tuned `p_n`/`r_c` if needed; a first dropout sweep |
| M11 | Analysis toolkit | All §6 tools run on M10 policies; RQ1–RQ2 first answers |
| M12 | Dropout training | Dropout policy multi-seed; RQ3–RQ5 first answers |
| M13 | Policy freeze | Any §7 iterations are recorded; frozen GQN checkpoints are named in the journal |
| M14 | Final experiments | All RQs on frozen policies, statistics; figures regenerate from `experiments/` |
| M15 | Release | Paper draft, public repo, README results table, demos |

---

## 11. Non-goals for v1

Everything in [`SPEC_v2.md`](SPEC_v2.md), unless a module is needed to answer a reviewer and is run as a clearly scoped check. That includes failure prediction, the MAPPO comparison, reinforcements, and team-size and behaviour generalisation.

---

## 12. Open items

| Item | Resolved in |
| --- | --- |
| Whether full training is affordable on available compute, and the resulting seed count | M10 |
| Dropout rates and timing grid that make a meaningful sweep | M9–M10 |
| Which embeddings to probe | M11 |
| Target venue | M13, once results exist |
