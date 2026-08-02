# Literature and novelty notes

## Primary predecessor

Wanqiao Xu, Yifan Zhu, and Benjamin Van Roy, “An Information-Theoretic
Definition for Open-Ended Learning,” arXiv:2606.08369, 2026.

The source paper defines the bit-equivalent and average bit-equivalent, proves
non-open-endedness for several classical bandits, constructs an open-ended
infinite-dimensional bandit, and proves success of a scheduled truncated
Thompson sampler. Version 1 does not state general reward-transformation,
reward-sufficient-source, behavioral-quotient, or frontier-composition
theorems. Its bounded logistic example shows that bounded nonlinear reward
does not by itself preclude open-endedness.

Primary source: <https://arxiv.org/abs/2606.08369>

## Classical foundations that constrain novelty claims

### Rate--distortion theory

The frontier is an inverse rate--distortion problem with negative utility in
place of distortion. Convexity, finite-alphabet existence, Lagrange duality,
and product-source arguments have classical antecedents. Shannon's 1959 paper
already gives independent-source infimal convolution and the corresponding
equal-slope allocation rule. These must be cited as foundations rather than
presented as wholly new information theory.

- C. E. Shannon, “Coding Theorems for a Discrete Source With a Fidelity
  Criterion,” 1959: <https://ieeexplore.ieee.org/document/5311476>
- I. Csiszár, “On an Extremum Problem of Information Theory,” 1974.

The same mutual-information/expected-utility optimization also appears in
rational inattention and information-theoretic bounded rationality.

- T. Genewein, F. Leibfried, J. Grau-Moya, and D. A. Braun, “Bounded
  Rationality, Abstraction, and Hierarchical Decision-Making,” 2015:
  <https://doi.org/10.3389/frobt.2015.00027>
- C. Sims, “Implications of Rational Inattention,” 2003:
  <https://pages.stern.nyu.edu/~dbackus/Exotic/1Robustness/Sims%20inattention%20JME%2003.pdf>
- F. Matějka and A. McKay, “Rational Inattention to Discrete Choices,” 2015:
  <https://pubs.aeaweb.org/doi/10.1257/aer.20130047>
- P. A. Ortega and D. A. Braun, “Information, Utility & Bounded Rationality,”
  2011: <https://arxiv.org/abs/1107.5766>

### Statistical experiments and sufficient representations

Data-processing proofs for source reduction and action garbling are closely
related to classical sufficiency and Blackwell comparison. The paper's likely
contribution is to identify the exact semantic quotient required by the new
open-endedness definition, not to claim discovery of data processing.

- D. Blackwell, “Comparison of Experiments,” 1951:
  <https://digicoll.lib.berkeley.edu/record/112749>
- D. Blackwell, “Equivalent Comparisons of Experiments,” 1953:
  <https://projecteuclid.org/euclid.aoms/1177729032>

### Expected utility and reward transformations

Uniqueness of expected-utility representations up to positive-affine
transformation is classical. Exact affine frontier invariance and the Jensen
maximality proof should be positioned as an application to bit-equivalent.
The potentially new result is an explicit open-endedness classification flip
under an invertible monotone reward transformation.

- J. von Neumann and O. Morgenstern, *Theory of Games and Economic Behavior*:
  <https://www.jstor.org/stable/j.ctt1r2gkx>

### Tail escape and flash signaling

The rare-burst construction resembles on--off or “flash” signaling: a
vanishing duty cycle is paired with increasing amplitude. The analogy helps
name the mechanism, but it is not the same theorem.

- S. Verdú, “On Channel Capacity per Unit Cost,” 1990:
  <https://doi.org/10.1109/18.57201>
- S. Verdú, “Spectral Efficiency in the Wideband Regime,” 2002:
  <https://web.mit.edu/6.441/www/reading/verdu_wideband.pdf>

### Decision-focused rate--distortion

Prior work uses rate--distortion to choose useful learning targets and compress
environments while retaining decision value. It is conceptually adjacent to
reward-sufficient reduction.

- D. Arumugam and B. Van Roy, “Deciding What to Learn: A Rate-Distortion
  Approach,” ICML 2021: <https://proceedings.mlr.press/v139/arumugam21a.html>
- D. Arumugam and B. Van Roy, “Deciding What to Model: Value-Equivalent
  Sampling for Reinforcement Learning,” 2022:
  <https://arxiv.org/abs/2206.02072>

## Current novelty posture

Likely classical or elementary:

- positive-affine remapping;
- convexity of the finite frontier;
- finite independent-product infimal convolution;
- reward-sufficient source reduction;
- action-quotient equality with a measurable lift;
- the Pinsker lower bound for bounded reward;
- compactness-based attainment;
- finite product composition and identical tensorization;
- Bernoulli time-sharing itself.

Potentially substantive when assembled and strengthened:

- positive-affine maps as the maximal *universal frontier-invariance* class;
- an invertible monotone reward transform that flips open-endedness while
  preserving feedback information;
- a taxonomy separating attained zero information, boundary nonattainment,
  positive-gap collapse, and infeasibility;
- the necessity of uncontrolled tails for vanishing-information sequences at
  a fixed positive reward gap, interpreted for the bit-equivalent;
- strict-margin counterexamples showing that a negative uninformed baseline
  alone does not prevent collapse;
- a rigorously delimited countable-product theorem connecting the frontier to
  the local information price of reward;
- consequences showing exactly which extra axioms are required before
  bit-equivalent can be interpreted as an intrinsic capability measure.

The safest positioning is a synthesis of classical rate--distortion,
expected-utility, and statistical-experiment theory that exposes previously
unreported robustness failures of the 2026 bit-equivalent/open-endedness
definition. The completed primary-source search and claim-level risk assessment
are in [`systematic-literature-audit.md`](systematic-literature-audit.md).
Priority claims remain provisional pending expert mathematical review and
citation chaining.
