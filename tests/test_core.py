from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys

import pytest

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec, additive_reward
from infinite_rulebook.core.rng import CounterRNG
from infinite_rulebook.environments.independent import IndependentRulebook


def test_baseline_reward_has_strict_negative_uninformed_margin() -> None:
    spec = RewardSpec()
    assert spec.profitability_threshold == 0.5
    assert spec.uninformed_reward == -0.5
    assert spec.contribution(0, 1) == 0
    assert spec.contribution(1, 1) == 1
    assert spec.contribution(2, 1) == -1


@pytest.mark.parametrize(
    ("q", "u", "c"),
    [(2, 1.0, 1.0), (4, 1.0, 1.0 / 3.0), (4, 0.0, 1.0)],
)
def test_reward_rejects_non_strict_or_invalid_margins(
    q: int, u: float, c: float
) -> None:
    with pytest.raises(ValueError):
        RewardSpec(q=q, u=u, c=c)


def test_reward_parameterization_properties() -> None:
    for q in range(2, 12):
        for u in (0.25, 1.0, 7.5):
            threshold = 1.0 / q + (1.0 - 1.0 / q) / 3.0
            c = u * threshold / (1.0 - threshold)
            spec = RewardSpec(q=q, u=u, c=c)
            assert spec.profitability_threshold > 1.0 / q
            assert spec.uninformed_reward < 0.0


@pytest.mark.parametrize("field", ["u", "c"])
def test_reward_rejects_boolean_values(field: str) -> None:
    arguments = {"q": 4, "u": 1.0, "c": 1.0, field: True}
    with pytest.raises(ValueError):
        RewardSpec(**arguments)


def test_action_canonicalizes_order_and_abstentions() -> None:
    left = DeploymentAction([(9, 2), (4, 0), (1, 3)])
    right = DeploymentAction([(1, 3), (9, 2)])
    assert left == right
    assert hash(left) == hash(right)
    assert left.entries == ((1, 3), (9, 2))
    assert left.support == (1, 9)
    assert left.prediction_for(4) == 0
    assert left.prediction_for(9) == 2


def test_all_serialization_orders_have_one_behavior() -> None:
    entries = [(100, 1), (3, 4), (17, 2), (8, 3)]
    actions = {DeploymentAction(order) for order in itertools.permutations(entries)}
    assert actions == {DeploymentAction(entries)}


@pytest.mark.parametrize(
    "entries",
    [
        [(1, 1), (1, 2)],
        [(1, 0), (1, 2)],
        [(0, 1)],
        [(-1, 1)],
        [(1, -1)],
        [(True, 1)],
        [(1, True)],
    ],
)
def test_action_rejects_ambiguous_or_invalid_entries(
    entries: list[tuple[int, int]],
) -> None:
    with pytest.raises(ValueError):
        DeploymentAction(entries)


def test_action_validates_predictions_against_environment_alphabet() -> None:
    action = DeploymentAction([(1, 5)])
    with pytest.raises(ValueError):
        action.validate_alphabet(4)


def test_counter_rng_is_typed_and_counter_deterministic() -> None:
    rng = CounterRNG("same seed", stream="test")
    assert rng.uint64(7, "x") == rng.uint64(7, "x")
    assert rng.uint64(7, "x") != rng.uint64("7", "x")
    assert rng.uint64(7, "x") != rng.uint64(7, b"x")
    assert rng.uint64(7, "x") != CounterRNG("same seed", "other").uint64(7, "x")


def test_lazy_labels_are_invariant_to_query_order() -> None:
    indices = (1, 2, 3, 19, 10**6, 2**80)
    forward = IndependentRulebook(90210)
    reverse = IndependentRulebook(90210)
    expected = {index: forward.label(index) for index in indices}
    observed = {index: reverse.label(index) for index in reversed(indices)}
    assert observed == expected
    assert all(1 <= label <= 4 for label in observed.values())


def test_lazy_labels_are_invariant_to_python_hash_seed() -> None:
    script = (
        "from infinite_rulebook.environments.independent import IndependentRulebook;"
        "e=IndependentRulebook('stable');"
        "print([e.label(i) for i in (1,2,10,999999)])"
    )
    outputs = []
    for hash_seed in ("1", "8675309"):
        environment = os.environ | {
            "PYTHONHASHSEED": hash_seed,
            "PYTHONPATH": "src",
        }
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=os.getcwd(),
                env=environment,
                text=True,
            )
        )
    assert outputs[0] == outputs[1]


def test_independent_environment_exact_reward_and_no_mutation() -> None:
    environment = IndependentRulebook(seed=123)
    indices = (1, 7, 200)
    labels = environment.labels(indices)
    action = DeploymentAction(
        [
            (indices[0], labels[0]),
            (indices[1], labels[1] % 4 + 1),
            (indices[2], labels[2]),
        ]
    )
    assert environment.evaluate(action) == 1.0
    assert environment.evaluate(action) == 1.0
    assert environment.labels(indices) == labels


def test_additive_reward_counts_once_per_supported_rule() -> None:
    labels = {1: 2, 2: 3, 9: 4}
    action = DeploymentAction([(9, 1), (1, 2), (2, 0)])
    spec = RewardSpec(q=4, u=2.0, c=1.0)
    assert additive_reward(action, labels.__getitem__, spec) == 1.0
    assert math.isclose(spec.from_counts(correct=1, incorrect=1), 1.0)
