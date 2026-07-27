"""Tests for replay-derived symbolic information telemetry."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from infinite_rulebook.information import InformationCategory
from infinite_rulebook.orchestration.config import EnvironmentKind
from infinite_rulebook.orchestration.telemetry import (
    ObservationNamespace,
    ReplayQueryObservation,
    information_breakdown_from_observations,
    information_ledger_from_observations,
)


def _observation(
    rule_index: int,
    value: int,
    *,
    round_index: int = 0,
    query_ordinal: int = 0,
    namespace: ObservationNamespace = ObservationNamespace.REWARD,
) -> ReplayQueryObservation:
    return ReplayQueryObservation(
        round_index,
        query_ordinal,
        namespace,
        rule_index,
        value,
    )


@pytest.mark.parametrize(
    "environment",
    (
        EnvironmentKind.IND,
        EnvironmentKind.ALEA,
        EnvironmentKind.PUBLIC_C,
    ),
)
def test_independent_reward_primitives_are_reward_relevant(
    environment: EnvironmentKind,
) -> None:
    breakdown = information_breakdown_from_observations(
        (_observation(7, 3),),
        environment=environment,
        q=4,
        epsilon=0.0,
    )

    assert breakdown.reward_relevant_nats == pytest.approx(math.log(4))
    assert breakdown.shared_core_nats == 0.0
    assert breakdown.total_acquired_nats == pytest.approx(math.log(4))


def test_red_aliases_invert_offsets_and_share_one_core_posterior() -> None:
    ledger = information_ledger_from_observations(
        (
            _observation(1, 2),
            _observation(3, 3, round_index=1),
        ),
        environment=EnvironmentKind.RED_C,
        q=4,
        epsilon=0.1,
        core_dimensions=2,
    )

    assert len(ledger.blocks) == 1
    block = ledger.blocks[0]
    assert block.axes[0].category is InformationCategory.SHARED_CORE
    assert tuple(item.surface_rule_id for item in block.surface_dependencies) == (
        "RED-C:surface:1",
        "RED-C:surface:3",
    )
    assert block.posterior[1] == max(block.posterior)
    assert ledger.breakdown.reward_relevant_nats == 0.0
    assert ledger.breakdown.shared_core_nats > 0.0


def test_mix_separates_odd_primitives_and_deduplicates_even_core_aliases() -> None:
    ledger = information_ledger_from_observations(
        (
            _observation(1, 4),
            _observation(2, 2, round_index=1),
            _observation(6, 3, round_index=2),
        ),
        environment=EnvironmentKind.MIX,
        q=4,
        epsilon=0.1,
        core_dimensions=2,
    )

    assert len(ledger.blocks) == 2
    categories = {block.axes[0].category for block in ledger.blocks}
    assert categories == {
        InformationCategory.REWARD_RELEVANT,
        InformationCategory.SHARED_CORE,
    }
    core = next(
        block
        for block in ledger.blocks
        if block.axes[0].category is InformationCategory.SHARED_CORE
    )
    assert len(core.surface_dependencies) == 2
    assert core.posterior[1] == max(core.posterior)
    assert ledger.breakdown.reward_relevant_nats > 0.0
    assert ledger.breakdown.shared_core_nats > 0.0


def test_trivia_is_persistent_distractor_information() -> None:
    ledger = information_ledger_from_observations(
        (
            _observation(1, 2),
            _observation(
                1,
                3,
                query_ordinal=1,
                namespace=ObservationNamespace.TRIVIA,
            ),
        ),
        environment=EnvironmentKind.TRIVIA,
        q=4,
        epsilon=0.0,
    )

    assert len(ledger.blocks) == 2
    breakdown = ledger.breakdown
    assert breakdown.reward_relevant_nats == pytest.approx(math.log(4))
    assert breakdown.persistent_distractor_nats == pytest.approx(math.log(4))
    assert breakdown.total_acquired_nats == pytest.approx(2 * math.log(4))


def test_alea_cosmetic_observations_are_ignored() -> None:
    reward = _observation(1, 2)
    cosmetic = _observation(
        1,
        197,
        namespace=ObservationNamespace.COSMETIC,
    )

    with_cosmetic = information_ledger_from_observations(
        (reward, cosmetic),
        environment=EnvironmentKind.ALEA,
        q=4,
        epsilon=0.1,
    )
    reward_only = information_ledger_from_observations(
        (reward,),
        environment=EnvironmentKind.ALEA,
        q=4,
        epsilon=0.1,
    )

    assert with_cosmetic == reward_only


def test_records_are_immutable_and_duplicate_replay_events_are_rejected() -> None:
    observation = _observation(1, 2)
    with pytest.raises(FrozenInstanceError):
        observation.value = 3  # type: ignore[misc]

    with pytest.raises(ValueError, match="duplicate replay observation"):
        information_ledger_from_observations(
            (observation, observation),
            environment=EnvironmentKind.IND,
            q=4,
            epsilon=0.1,
        )


def test_ledger_is_independent_of_replay_iteration_order() -> None:
    observations = (
        _observation(1, 2),
        _observation(2, 4, round_index=1),
    )

    forward = information_ledger_from_observations(
        observations,
        environment=EnvironmentKind.IND,
        q=4,
        epsilon=0.1,
    )
    reverse = information_ledger_from_observations(
        reversed(observations),
        environment=EnvironmentKind.IND,
        q=4,
        epsilon=0.1,
    )

    assert forward == reverse


def test_namespaces_are_environment_specific() -> None:
    trivia = _observation(1, 2, namespace=ObservationNamespace.TRIVIA)
    cosmetic = _observation(1, 0, namespace=ObservationNamespace.COSMETIC)

    with pytest.raises(ValueError, match="TRIVIA"):
        information_ledger_from_observations(
            (trivia,),
            environment=EnvironmentKind.IND,
            q=4,
            epsilon=0.1,
        )
    with pytest.raises(ValueError, match="ALEA"):
        information_ledger_from_observations(
            (cosmetic,),
            environment=EnvironmentKind.IND,
            q=4,
            epsilon=0.1,
        )
