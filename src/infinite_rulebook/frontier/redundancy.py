"""Analytic witnesses and bounds for finite-core redundancy."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from infinite_rulebook.core.behavior import DeploymentAction
from infinite_rulebook.core.reward import RewardSpec
from infinite_rulebook.environments.redundant import cyclic_surface_map
from infinite_rulebook.frontier.finite_problem import FiniteDecisionProblem
from infinite_rulebook.frontier.rulebook_problem import EnumeratedRulebookProblem


def _real_not_nan(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} must not be NaN")
    return result


def _finite_nonnegative(value: Real, name: str) -> float:
    result = _real_not_nan(value, name)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _reward_spec(value: RewardSpec | None) -> RewardSpec:
    if value is None:
        return RewardSpec()
    if not isinstance(value, RewardSpec):
        raise TypeError("reward_spec must be a RewardSpec")
    return value


def _validate_enumeration_size(
    state_count: int,
    action_count: int,
    max_matrix_entries: int,
) -> None:
    limit = _positive_integer(max_matrix_entries, "max_matrix_entries")
    if state_count * action_count > limit:
        raise ValueError(
            "enumerated reward matrix would exceed max_matrix_entries: "
            f"{state_count} * {action_count}"
        )


def _capped_actions(
    indices: tuple[int, ...],
    q: int,
    cap: int,
) -> tuple[DeploymentAction, ...]:
    actions = []
    for support_size in range(cap + 1):
        for support in itertools.combinations(indices, support_size):
            actions.extend(
                DeploymentAction(zip(support, predictions, strict=True))
                for predictions in itertools.product(
                    range(1, q + 1),
                    repeat=support_size,
                )
            )
    return tuple(actions)


def _derived_label(
    core: tuple[int, ...],
    derived_index: int,
    q: int,
) -> int:
    component, offset = cyclic_surface_map(derived_index, len(core), q)
    return 1 + ((core[component] - 1 + offset) % q)


def enumerate_redundant_rulebook(
    core_dimensions: int,
    derived_rules: int,
    max_derived_support: int | None,
    reward_spec: RewardSpec | None = None,
    *,
    max_matrix_entries: int = 2_000_000,
) -> EnumeratedRulebookProblem:
    """Enumerate a finite RED projection for exact channel optimization.

    ``None`` removes the support cap only within this finite projection. It
    does not represent or approximate the exact infinite RED-U frontier.
    """

    dimensions = _positive_integer(core_dimensions, "core_dimensions")
    rules = _nonnegative_integer(derived_rules, "derived_rules")
    if max_derived_support is None:
        cap = rules
    else:
        cap = min(
            _nonnegative_integer(max_derived_support, "max_derived_support"),
            rules,
        )
    spec = _reward_spec(reward_spec)
    state_count = spec.q**dimensions
    action_count = sum(
        math.comb(rules, support) * spec.q**support for support in range(cap + 1)
    )
    _validate_enumeration_size(
        state_count,
        action_count,
        max_matrix_entries,
    )

    states = tuple(itertools.product(range(1, spec.q + 1), repeat=dimensions))
    actions = _capped_actions(tuple(range(1, rules + 1)), spec.q, cap)
    rewards = tuple(
        tuple(
            math.fsum(
                spec.contribution(
                    prediction,
                    _derived_label(state, index, spec.q),
                )
                for index, prediction in action
            )
            for action in actions
        )
        for state in states
    )
    prior = (1.0 / state_count,) * state_count
    return EnumeratedRulebookProblem(
        problem=FiniteDecisionProblem(prior=prior, rewards=rewards),
        states=states,
        actions=actions,
    )


def enumerate_mixed_rulebook(
    primitive_dimensions: int,
    core_dimensions: int,
    derived_rules: int,
    max_redundant_support: int | None,
    reward_spec: RewardSpec | None = None,
    *,
    max_matrix_entries: int = 2_000_000,
) -> EnumeratedRulebookProblem:
    """Enumerate MIX with odd primitives and capped even derived rules.

    A state stores its primitive coordinates first and its finite redundant
    core second. The cap applies only to even-indexed predictions.
    """

    primitives = _nonnegative_integer(
        primitive_dimensions,
        "primitive_dimensions",
    )
    dimensions = _positive_integer(core_dimensions, "core_dimensions")
    rules = _nonnegative_integer(derived_rules, "derived_rules")
    if primitives + rules < 1:
        raise ValueError("MIX must contain at least one projected rule")
    if max_redundant_support is None:
        cap = rules
    else:
        cap = min(
            _nonnegative_integer(
                max_redundant_support,
                "max_redundant_support",
            ),
            rules,
        )
    spec = _reward_spec(reward_spec)
    state_count = spec.q ** (primitives + dimensions)
    redundant_action_count = sum(
        math.comb(rules, support) * spec.q**support for support in range(cap + 1)
    )
    action_count = (spec.q + 1) ** primitives * redundant_action_count
    _validate_enumeration_size(
        state_count,
        action_count,
        max_matrix_entries,
    )

    states = tuple(
        itertools.product(
            range(1, spec.q + 1),
            repeat=primitives + dimensions,
        )
    )
    derived_actions = _capped_actions(
        tuple(range(2, 2 * rules + 1, 2)),
        spec.q,
        cap,
    )
    actions = tuple(
        DeploymentAction(
            [
                (2 * primitive + 1, prediction)
                for primitive, prediction in enumerate(primitive_vector)
            ]
            + list(derived_action)
        )
        for primitive_vector in itertools.product(
            range(0, spec.q + 1),
            repeat=primitives,
        )
        for derived_action in derived_actions
    )
    rewards = tuple(
        tuple(
            math.fsum(
                spec.contribution(
                    prediction,
                    (
                        state[(index - 1) // 2]
                        if index % 2
                        else _derived_label(
                            state[primitives:],
                            index // 2,
                            spec.q,
                        )
                    ),
                )
                for index, prediction in action
            )
            for action in actions
        )
        for state in states
    )
    prior = (1.0 / state_count,) * state_count
    return EnumeratedRulebookProblem(
        problem=FiniteDecisionProblem(prior=prior, rewards=rewards),
        states=states,
        actions=actions,
    )


@dataclass(frozen=True, slots=True)
class RareBurstWitness:
    """A stochastic RED-U channel that bursts only after revealing the core.

    With probability ``target_reward / (support * correct_reward)`` the
    channel emits ``support`` correct derived predictions; otherwise it
    abstains. Its information is at most the reveal probability times the
    core entropy.
    """

    target_reward: float
    support: int
    core_entropy: float
    correct_reward: float = 1.0

    def __post_init__(self) -> None:
        target = _finite_nonnegative(self.target_reward, "target_reward")
        support = _positive_integer(self.support, "support")
        entropy = _finite_nonnegative(self.core_entropy, "core_entropy")
        reward = _finite_nonnegative(self.correct_reward, "correct_reward")
        if not math.isfinite(target):
            raise ValueError("target_reward must be finite")
        if not math.isfinite(entropy):
            raise ValueError("core_entropy must be finite")
        if not math.isfinite(reward) or reward == 0.0:
            raise ValueError("correct_reward must be finite and positive")
        if target > support * reward:
            raise ValueError("support is too small to attain target_reward")
        object.__setattr__(self, "target_reward", target)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "core_entropy", entropy)
        object.__setattr__(self, "correct_reward", reward)

    @property
    def deployment_probability(self) -> float:
        return self.target_reward / (self.support * self.correct_reward)

    @property
    def expected_reward(self) -> float:
        return self.deployment_probability * self.support * self.correct_reward

    @property
    def expected_support(self) -> float:
        """Expected number of deployed rules in the rare-burst channel."""

        return self.deployment_probability * self.support

    @property
    def information_upper_bound(self) -> float:
        """Upper bound on ``I(core; action)`` in nats."""

        return self.deployment_probability * self.core_entropy


def rare_burst_sequence(
    target_reward: Real,
    supports: Iterable[int],
    *,
    core_dimensions: int,
    q: int,
    correct_reward: Real = 1.0,
) -> tuple[RareBurstWitness, ...]:
    """Construct a finite sequence witnessing the RED-U limit."""

    dimensions = _positive_integer(core_dimensions, "core_dimensions")
    alphabet = _positive_integer(q, "q")
    if alphabet < 2:
        raise ValueError("q must be at least two")
    target = _finite_nonnegative(target_reward, "target_reward")
    reward = _finite_nonnegative(correct_reward, "correct_reward")
    if reward == 0.0:
        raise ValueError("correct_reward must be positive")
    entropy = dimensions * math.log(alphabet)
    return tuple(
        RareBurstWitness(
            target_reward=target,
            support=support,
            core_entropy=entropy,
            correct_reward=reward,
        )
        for support in supports
    )


def unrestricted_redundant_bit_equivalent(target_reward: Real) -> float:
    """Return the exact RED-U frontier: zero at every finite threshold."""

    target = _real_not_nan(target_reward, "target_reward")
    if target <= 0.0:
        return 0.0
    return math.inf if math.isinf(target) else 0.0


def capped_redundant_information_upper_bound(
    target_reward: Real,
    *,
    max_derived_support: int,
    core_dimensions: int,
    q: int,
    correct_reward: Real = 1.0,
) -> float:
    """Rare-burst upper bound for RED-C, or infinity when infeasible."""

    target = _real_not_nan(target_reward, "target_reward")
    support = _positive_integer(max_derived_support, "max_derived_support")
    dimensions = _positive_integer(core_dimensions, "core_dimensions")
    alphabet = _positive_integer(q, "q")
    if alphabet < 2:
        raise ValueError("q must be at least two")
    reward = _finite_nonnegative(correct_reward, "correct_reward")
    if not math.isfinite(reward) or reward == 0.0:
        raise ValueError("correct_reward must be finite and positive")
    if target <= 0.0:
        return 0.0
    if target > support * reward:
        return math.inf
    return (target / (support * reward)) * dimensions * math.log(alphabet)
