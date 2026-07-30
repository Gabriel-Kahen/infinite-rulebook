"""Bounded approximate estimators and synthetic calibration tools."""

from infinite_rulebook.estimators.behavioral import (
    BehavioralEstimatorConfig,
    BehavioralFit,
    BehavioralFrontierEstimate,
    BehavioralFrontierPoint,
    IdentificationStatus,
    estimate_behavioral_frontier,
    fit_behavioral_channel,
)
from infinite_rulebook.estimators.calibration import (
    CalibrationCase,
    CalibrationPoint,
    CalibrationReport,
    CalibrationSplit,
    CalibrationSummary,
    calibrate_behavioral_estimator,
)

__all__ = [
    "BehavioralEstimatorConfig",
    "BehavioralFit",
    "BehavioralFrontierEstimate",
    "BehavioralFrontierPoint",
    "CalibrationCase",
    "CalibrationPoint",
    "CalibrationReport",
    "CalibrationSplit",
    "CalibrationSummary",
    "IdentificationStatus",
    "calibrate_behavioral_estimator",
    "estimate_behavioral_frontier",
    "fit_behavioral_channel",
]
