# Theory paper research program

## Working title

**When Does Reward Require Information? Invariances and Degeneracies of the
Bit-Equivalent**

## Central question

When is the bit-equivalent an intrinsic, nondegenerate measure of
reward-relevant capability rather than an artifact of reward semantics,
parameterization, action representation, or rare-event randomization?

## Intended contribution

The paper will study the mathematical object

\[
B_\rho
=
\inf_{P(A\mid\Theta):\mathbb E[r_\Theta(A)]\geq\rho}
I(\Theta;A)
\]

as a reward--information frontier. It will not present a benchmark evaluation
or claim experimental validation.

The target theorem package is:

1. **Reward semantics:** exact positive-affine invariance, a maximality result
   for transformations preserving expected-reward frontiers universally, and
   an explicit invertible monotone transformation that flips an environment
   from open-ended to non-open-ended without removing feedback information.
2. **Representation semantics:** exact reduction to reward-sufficient source
   statistics and exact quotienting of behaviorally equivalent actions when an
   admissible source-independent lift exists.
3. **Nondegeneracy:** a quantitative bounded-reward lower bound and a
   necessary loss-of-uniform-integrability condition for positive-gap
   nonattained collapse.
4. **Composition:** an infimal-convolution law for independent additive
   problems, with finite tensorization and a countable-product local-slope
   theorem under an explicit finite-expected-support contract.

## Why these four belong together

The first two results identify transformations that leave the decision problem
unchanged. The third identifies conditions under which the frontier genuinely
prices information instead of being defeated by rare high-reward lotteries.
The fourth explains how local reward--information costs accumulate into a
large or open-ended problem. Together they answer whether the bit-equivalent is
well defined, nondegenerate, and compositional.

## Proposed paper structure

1. Introduction: reward-relevant information needs a foundations layer.
2. Bayesian decision problems and reward--information frontiers.
3. Invariance to sufficient sources and behavioral action quotients.
4. Reward transformations: affine invariance and nonlinear relativity.
5. Nondegeneracy: bounded rewards, zero-information baselines, and rare bursts.
6. Independent composition and the local information price of reward.
7. Consequences for definitions of open-ended learning.
8. Limits and open problems: priors, sequential state, identification, and
   schedule-free agents.

## Scope exclusions

- No experimental result is needed for a theorem.
- Existing calibration and confirmatory-study material is not part of the
  theory paper's evidentiary argument.
- Numerical solvers may be used to check examples, but they are not a paper
  contribution unless they expose a mathematical conjecture.
- Stateful and nonstationary extensions, estimation from black-box data, and
  adaptive learning-target algorithms remain separate projects unless the
  four-part theorem package becomes too small after the novelty audit.

## Decision gates

Before drafting the full manuscript:

1. Complete proof-level statements for the finite setting.
2. Determine which results are standard rate--distortion corollaries and which
   are new in the open-ended-learning setting.
3. Produce at least one classification-changing nonlinear reward
   counterexample and one nonattained zero-frontier example. The prefix-query
   construction with the cubic reward transform currently supplies the first.
4. Generalize at least one result beyond identical independent coordinates;
   otherwise the composition contribution is too narrow.
5. Obtain an adversarial mathematical review of every theorem.
