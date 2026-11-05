"""Policy contracts: uncapped graphs, synchronous gradients and team Bellman targets."""

import numpy as np
import swarmsim
import torch
from swarm_policy.model import GQN, tensors
from swarm_policy.replay import NStep, Transition, pack, split
from swarm_policy.train import TrainConfig, double_dqn_target, team_sum, update


def graph():
    e = swarmsim.TeamEnv(swarmsim.EnvConfig.load("scenarios/v0/env.yaml"))
    g = e.reset(123)
    for _ in range(100):
        g = e.step([13] * len(g["self"]))["observation"]
    return g


def test_empty_nodes_and_empty_team():
    torch.set_num_threads(1)
    g = graph()
    model = GQN()
    q, *_ = model(*tensors(g))
    assert q.shape == (3, 27)
    empty = split(g, 99)
    q, *_ = model(*tensors(empty))
    assert q.shape == (0, 27)
    assert torch.isfinite(model(*tensors(g))[0]).all()


def test_permutation_and_message_gradients():
    g = graph()
    model = GQN()
    q = model(*tensors(g))[0]
    shuffled = {k: v.copy() for k, v in g.items()}
    rng = np.random.default_rng(2)
    for kind, owner in (
        ("local_threat", "local_owner"),
        ("global_threat", "global_owner"),
        ("teammate", "teammate_owner"),
    ):
        order = rng.permutation(len(g[kind]))
        shuffled[kind] = g[kind][order]
        shuffled[owner] = g[owner][order]
        if kind == "teammate":
            shuffled["teammate_src"] = g["teammate_src"][order]
    torch.testing.assert_close(q, model(*tensors(shuffled))[0], atol=1e-6, rtol=1e-6)
    q.sum().backward()
    assert model.message.weight.grad.abs().sum() > 0


def test_nstep_termination_and_truncation():
    g = graph()
    a = np.array([13] * len(g["self"]))
    n = NStep(3, 0.5)
    assert n.append(g, a, 1, g, False, False) == []
    r = n.append(g, a, 2, g, False, True)
    assert [x.reward for x in r] == [2, 2]
    assert [x.discount for x in r] == [0.25, 0.5]
    n = NStep(3, 0.5)
    n.append(g, a, 1, g, False, False)
    r = n.append(g, a, 2, g, True, False)
    assert [x.discount for x in r] == [0, 0]


def test_vdn_sum_and_learning_on_fixed_target():
    torch.manual_seed(3)
    torch.testing.assert_close(
        team_sum(torch.tensor([1.0, 2.0, 4.0]), torch.tensor([0, 0, 2]), 4),
        torch.tensor([3.0, 0.0, 4.0, 0.0]),
    )
    g = graph()
    dead = split(g, 99)
    batch = [Transition(g, np.array([13] * len(g["self"])), 1.0, dead, 0.0)]
    model = GQN()
    target = GQN()
    target.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    cfg = TrainConfig()
    losses = [update(model, target, opt, batch, cfg) for _ in range(60)]
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) * 0.1


def test_split_pack_roundtrip():
    g = graph()
    packed = pack([g, g])
    a = split(packed, 1)
    for key in g:
        np.testing.assert_array_equal(g[key], a[key])


def test_double_dqn_selection_death_and_empty_team():
    online = torch.tensor([[1.0, 3.0], [5.0, 2.0], [2.0, 4.0]])
    target = torch.tensor([[10.0, 7.0], [6.0, 20.0], [100.0, 9.0]])
    actual = double_dqn_target(
        online,
        target,
        torch.tensor([0, 0, 1]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.5, 0.0, 0.9]),
    )
    torch.testing.assert_close(actual, torch.tensor([7.5, 2.0, 3.0]))


def test_compressed_replay_ring_preserves_graphs():
    from swarm_policy.replay import Replay

    g = graph()
    replay = Replay(3, np.random.default_rng(1))
    for reward in range(5):
        replay.add(Transition(g, np.array([13] * len(g["self"])), float(reward), g, 0.9))
    assert len(replay) == 3
    for item in replay.sample(10):
        assert item.reward in (2, 3, 4)
        for key in g:
            np.testing.assert_array_equal(item.observation[key], g[key])
