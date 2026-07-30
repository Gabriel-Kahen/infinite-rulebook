# Learned-estimator foundation

This module is an engineering and synthetic-calibration foundation for the
direct behavioral channel estimator in the research plan. It is deliberately
bounded to small, fully enumerated `FiniteDecisionProblem` instances. It does
not use symbolic v2 study data, alter the frozen v2 design, or authorize an
approximate large-instance scientific claim.

The implementation is in `infinite_rulebook.estimators`.

## Reported object

For a finite prior, finite canonical action set, and reward matrix, the
estimator fits tabular channels on a prespecified nonnegative Lagrange
multiplier grid. Each fit alternates a Gibbs channel update with a learned
full-support action reference marginal for a fixed number of deterministic
steps. There is no random initialization or hidden stopping rule.

For a fitted channel \(q(a\mid\theta)\) and reference \(m(a)\), it reports

\[
\mathbb E_\Theta D_{\mathrm{KL}}(q(A\mid\Theta)\Vert m(A))
=
I(\Theta;A)
+
D_{\mathrm{KL}}(q(A)\Vert m(A)).
\]

The code independently evaluates all three terms. The reference quantity is
therefore an upper bound on that channel's mutual information. In this
finite-only foundation, the final reward and mutual information are also
computed exactly from the complete channel; no sampled confidence statement
is needed or implied.

For a requested reward \(\rho\):

- the upper endpoint is the mutual information of a retained, directly
  re-evaluated feasible channel whose reward clears \(\rho\);
- the lower endpoint is the best nonnegative bound
  \(\beta\rho+c_\beta(m)\) over the prespecified multiplier grid, where
  \(c_\beta(m)\) is the existing global finite-problem Lagrangian certificate;
- the interval is labeled `certified-partial-identification`;
- the zero-information region and targets above maximum reward are reported
  explicitly; and
- optimizer convergence is only a diagnostic. A lower certificate remains
  valid when the fixed optimization budget ends early.

The estimator never repairs, smooths, or narrows bounds to make a curve look
better. A lower certificate that exceeds a feasible witness beyond numerical
roundoff fails closed.

## Synthetic calibration

`calibrate_behavioral_estimator` compares estimated intervals with the
certified exact finite solver on named reward grids. Cases are tagged
`development` or `held-out` before evaluation. The report retains, per point:

- both estimated endpoints and the feasible upper witness;
- both endpoints of the certified exact envelope;
- whether the estimate contains that complete envelope;
- a signed interval for upper-bound excess;
- normalized achieved-reward overshoot; and
- exact-solver convergence.

Each split reports the fraction and count of prespecified grid points whose
certified exact envelopes are contained, maximum partial-identification width,
maximum upper excess, and maximum normalized reward overshoot. The grid
fraction is a descriptive diagnostic only. Target points and cases need not be
independent or exchangeable, so no binomial interval or population coverage
claim is reported. Development-case tuning can bias even the descriptive
fraction, and the function cannot enforce that an operator refrained from
looking at a held-out report before changing a later configuration.

```python
from infinite_rulebook.estimators import (
    BehavioralEstimatorConfig,
    CalibrationCase,
    CalibrationSplit,
    calibrate_behavioral_estimator,
)
from infinite_rulebook.frontier import one_coordinate_problem

config = BehavioralEstimatorConfig(
    betas=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
    optimizer_steps=128,
    maximum_states=32,
    maximum_actions=64,
)
report = calibrate_behavioral_estimator(
    (
        CalibrationCase(
            "q4-held-out",
            CalibrationSplit.HELD_OUT,
            one_coordinate_problem(q=4, u=1.0, c=1.0),
            (0.1, 0.5, 0.8),
        ),
    ),
    config=config,
)
```

## Scientific boundary and remaining work

The current implementation does **not** provide:

- an autoregressive canonical-rulebook channel;
- a declared finite projection, maximum index, maximum support, and stopping
  contract for learned rulebooks;
- Monte Carlo reward lower bounds or reference-KL upper confidence bounds;
- retained sampled channels and seed manifests;
- an \(N\)-convergence study;
- latent-bottleneck or learned converse estimators;
- calibrated distractor-leakage, inversion-slope, or scaling-classification
  gates; or
- evidence that synthetic coverage transfers to a larger action space.

The configured state and action caps enforce the small finite scope. Larger
or sampled applications require the missing statistical and artifact
contracts before they can be called achievable statistical upper bounds.
Until then, large-instance and asymptotic interpretation is prohibited.
