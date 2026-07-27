"""Reference learning agents."""

from infinite_rulebook.agents.comparison import (
    ExpandingTargetSchedule,
    FactorizedQueryAgent,
    FixedTargetPolicy,
    NoveltyDirectedPolicy,
    RelevantInformationDirectedPolicy,
    RewardDirectedPolicy,
    ScheduledTargetPolicy,
    TotalInformationDirectedPolicy,
    distractor_targets,
    useful_targets,
)
from infinite_rulebook.agents.protocols import (
    AcquisitionContext,
    AgentCheckpoint,
    CapabilityManifest,
    ObservationBatch,
    QueryAction,
    QueryTarget,
    SymbolicAgent,
    TargetKey,
)
from infinite_rulebook.agents.sanity import (
    FreshCoordinateSanityAgent,
    average_bit_equivalent,
    average_bit_equivalent_slope,
    bit_equivalent_slope,
    expected_coordinate_reward,
    expected_reward_slope,
)

__all__ = [
    "AcquisitionContext",
    "AgentCheckpoint",
    "CapabilityManifest",
    "ExpandingTargetSchedule",
    "FactorizedQueryAgent",
    "FixedTargetPolicy",
    "FreshCoordinateSanityAgent",
    "NoveltyDirectedPolicy",
    "ObservationBatch",
    "QueryAction",
    "QueryTarget",
    "RelevantInformationDirectedPolicy",
    "RewardDirectedPolicy",
    "ScheduledTargetPolicy",
    "SymbolicAgent",
    "TargetKey",
    "TotalInformationDirectedPolicy",
    "average_bit_equivalent",
    "average_bit_equivalent_slope",
    "bit_equivalent_slope",
    "distractor_targets",
    "expected_coordinate_reward",
    "expected_reward_slope",
    "useful_targets",
]
