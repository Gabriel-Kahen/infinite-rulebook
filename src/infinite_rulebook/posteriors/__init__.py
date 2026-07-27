"""Exact and approximate posterior representations."""

from infinite_rulebook.posteriors.categorical import (
    CategoricalPosterior,
    thresholded_deployment,
)

__all__ = ["CategoricalPosterior", "thresholded_deployment"]
