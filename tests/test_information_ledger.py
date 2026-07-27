"""Exact and adversarial tests for finite-closure information accounting."""

from __future__ import annotations

import math

import pytest

from infinite_rulebook.artifacts import scientific_payload_hash
from infinite_rulebook.information import (
    InformationBreakdown,
    InformationCategory,
    InformationLedger,
    LatentAxis,
    PosteriorBlock,
    SurfaceDependency,
    posterior_to_prior_kl,
)
from infinite_rulebook.posteriors import CategoricalPosterior


def test_exact_posterior_to_prior_kl_small_cases() -> None:
    assert posterior_to_prior_kl((0.25,) * 4, (0.25,) * 4) == pytest.approx(0.0)
    assert posterior_to_prior_kl((1.0, 0.0, 0.0, 0.0), (0.25,) * 4) == (
        pytest.approx(math.log(4))
    )
    assert posterior_to_prior_kl((0.5, 0.5), (0.25, 0.75)) == pytest.approx(
        0.5 * math.log(4.0 / 3.0)
    )
    assert math.isinf(posterior_to_prior_kl((0.5, 0.5), (1.0, 0.0)))
    assert posterior_to_prior_kl((1.0, 0.0), (5e-324, 1.0)) == pytest.approx(
        -math.log(5e-324)
    )


def test_sparse_touched_identifier_does_not_materialize_an_infinite_prefix() -> None:
    block = PosteriorBlock(
        block_id="single-touch",
        axes=(
            LatentAxis(
                "useful:1000000000000",
                4,
                InformationCategory.REWARD_RELEVANT,
            ),
        ),
        prior=(0.25,) * 4,
        posterior=(0.0, 0.0, 1.0, 0.0),
    )

    ledger = InformationLedger((block,))

    assert len(ledger.blocks) == 1
    assert len(ledger.blocks[0].posterior) == 4
    assert ledger.breakdown.total_acquired_nats == pytest.approx(math.log(4))
    assert ledger.validate().valid


def test_correlated_parity_information_uses_joint_chain_rule() -> None:
    block = PosteriorBlock(
        block_id="parity",
        axes=(
            LatentAxis("x", 2, InformationCategory.REWARD_RELEVANT),
            LatentAxis("y", 2, InformationCategory.SHARED_CORE),
        ),
        prior=(0.25, 0.25, 0.25, 0.25),
        posterior=(0.5, 0.0, 0.0, 0.5),
    )

    assert posterior_to_prior_kl((0.5, 0.5), (0.5, 0.5)) == pytest.approx(0.0)
    assert block.total_nats == pytest.approx(math.log(2))
    assert block.chain_contributions == pytest.approx((0.0, math.log(2)))
    assert InformationLedger((block,)).validate().valid


def test_canonical_axis_order_makes_relevant_projection_invariant() -> None:
    useful = LatentAxis("useful", 2, InformationCategory.REWARD_RELEVANT)
    distractor = LatentAxis("distractor", 2, InformationCategory.PERSISTENT_DISTRACTOR)
    forward = PosteriorBlock(
        "coupled",
        (useful, distractor),
        (0.25,) * 4,
        (0.5, 0.0, 0.0, 0.5),
    )
    reversed_input = PosteriorBlock(
        "coupled",
        (distractor, useful),
        (0.25,) * 4,
        (0.5, 0.0, 0.0, 0.5),
    )

    assert forward == reversed_input
    breakdown = InformationLedger((reversed_input,)).breakdown
    assert breakdown.reward_relevant_nats == pytest.approx(0.0)
    assert breakdown.persistent_distractor_nats == pytest.approx(math.log(2))


@pytest.mark.parametrize("surface_count", [0, 2, 100])
def test_red_surface_aliases_never_double_count_shared_core(
    surface_count: int,
) -> None:
    dependencies = tuple(
        SurfaceDependency(f"red:{index}", ("core:z",))
        for index in reversed(range(surface_count))
    )
    block = PosteriorBlock(
        block_id="red-core",
        axes=(LatentAxis("core:z", 2, InformationCategory.SHARED_CORE),),
        prior=(0.5, 0.5),
        posterior=(1.0, 0.0),
        surface_dependencies=dependencies,
    )

    breakdown = InformationLedger((block,)).breakdown

    assert breakdown.reward_relevant_nats == 0.0
    assert breakdown.shared_core_nats == pytest.approx(math.log(2))
    assert breakdown.relevant_nats == pytest.approx(math.log(2))
    assert breakdown.total_acquired_nats == pytest.approx(math.log(2))


def test_mix_separates_independent_growth_from_shared_core() -> None:
    useful = PosteriorBlock(
        "mix-independent",
        (LatentAxis("primitive:7", 2, InformationCategory.REWARD_RELEVANT),),
        (0.5, 0.5),
        (0.0, 1.0),
    )
    core = PosteriorBlock(
        "mix-core",
        (LatentAxis("core:1", 2, InformationCategory.SHARED_CORE),),
        (0.5, 0.5),
        (1.0, 0.0),
        tuple(
            SurfaceDependency(f"derived:{index}", ("core:1",)) for index in range(20)
        ),
    )

    breakdown = InformationLedger((useful, core)).breakdown

    assert breakdown.reward_relevant_nats == pytest.approx(math.log(2))
    assert breakdown.shared_core_nats == pytest.approx(math.log(2))
    assert breakdown.total_acquired_nats == pytest.approx(2 * math.log(2))
    assert breakdown.reconciles()


def test_trivia_changes_only_distractor_and_total_information() -> None:
    useful = PosteriorBlock(
        "useful",
        (LatentAxis("u", 2, InformationCategory.REWARD_RELEVANT),),
        (0.5, 0.5),
        (1.0, 0.0),
    )
    trivia = PosteriorBlock(
        "trivia",
        (LatentAxis("d", 2, InformationCategory.PERSISTENT_DISTRACTOR),),
        (0.5, 0.5),
        (0.0, 1.0),
    )

    without = InformationLedger((useful,)).breakdown
    with_trivia = InformationLedger((useful, trivia)).breakdown

    assert with_trivia.relevant_nats == pytest.approx(without.relevant_nats)
    assert with_trivia.persistent_distractor_nats == pytest.approx(math.log(2))
    assert with_trivia.total_acquired_nats == pytest.approx(
        without.total_acquired_nats + math.log(2)
    )


def test_ledger_rejects_duplicate_primitive_identity_across_blocks() -> None:
    axis = LatentAxis("same", 2, InformationCategory.REWARD_RELEVANT)
    first = PosteriorBlock("one", (axis,), (0.5, 0.5), (1.0, 0.0))
    second = PosteriorBlock("two", (axis,), (0.5, 0.5), (0.0, 1.0))

    with pytest.raises(ValueError, match="multiple blocks"):
        InformationLedger((first, second))


def test_semantic_collection_order_does_not_change_ledger_hash() -> None:
    first = PosteriorBlock(
        "a",
        (LatentAxis("x", 2, InformationCategory.REWARD_RELEVANT),),
        (0.5, 0.5),
        (1.0, 0.0),
        (
            SurfaceDependency("rule:2", ("x",)),
            SurfaceDependency("rule:1", ("x",)),
        ),
    )
    second = PosteriorBlock(
        "b",
        (LatentAxis("z", 2, InformationCategory.SHARED_CORE),),
        (0.5, 0.5),
        (0.0, 1.0),
    )

    forward = InformationLedger((first, second))
    reverse = InformationLedger((second, first))

    assert forward == reverse
    assert scientific_payload_hash(forward) == scientific_payload_hash(reverse)


def test_absolute_continuity_violation_is_an_explicit_diagnostic() -> None:
    block = PosteriorBlock(
        "bad-support",
        (LatentAxis("z", 2, InformationCategory.SHARED_CORE),),
        (1.0, 0.0),
        (0.5, 0.5),
    )

    report = InformationLedger((block,)).validate()

    assert not report.valid
    assert tuple(item.code for item in report.diagnostics) == (
        "ABSOLUTE_CONTINUITY_VIOLATION",
    )


def test_information_records_are_immutable() -> None:
    axis = LatentAxis("z", 2, InformationCategory.SHARED_CORE)

    with pytest.raises(AttributeError):
        axis.cardinality = 3  # type: ignore[misc]


def test_probability_and_breakdown_artifacts_reject_malformed_values() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        posterior_to_prior_kl((8.0, 2.0), (2.0, 2.0))
    with pytest.raises(ValueError, match="cannot be negative"):
        InformationBreakdown(-1.0, 2.0, 0.0, 0.0, 0.0, 1.0)


def test_negative_approximation_total_returns_diagnostic_instead_of_raising() -> None:
    report = InformationLedger((), approximation_residual_nats=-1.0).validate()

    assert not report.valid
    assert tuple(item.code for item in report.diagnostics) == (
        "NEGATIVE_TOTAL_INFORMATION",
    )


def test_expected_realized_kl_equals_population_mutual_information() -> None:
    q = 3
    epsilon = 0.2
    history_probabilities = (1.0 / q,) * q
    realized_surprises = []
    for observation in range(1, q + 1):
        posterior = CategoricalPosterior(q=q, epsilon=epsilon)
        posterior.update(observation)
        realized_surprises.append(posterior.kl_from_prior)
    expected_surprise = math.fsum(
        probability * surprise
        for probability, surprise in zip(
            history_probabilities, realized_surprises, strict=True
        )
    )
    channel_mutual_information = (
        math.log(q)
        + (1.0 - epsilon) * math.log(1.0 - epsilon)
        + epsilon * math.log(epsilon / (q - 1))
    )

    assert expected_surprise == pytest.approx(channel_mutual_information)
