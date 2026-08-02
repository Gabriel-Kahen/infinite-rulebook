# External mathematical review brief

## Status and purpose

This packet is ready to send to an information theorist or mathematical
decision theorist. It records the exact questions on which the draft needs an
independent human judgment. Creating this packet does **not** mean that an
external review has occurred.

The object under study is the reward--information frontier

\[
B_{\mathsf D}(\rho)
=
\inf_{P(A\mid\Theta):\mathbb E[r(\Theta,A)]\geq\rho}I(\Theta;A).
\]

The manuscript uses this static frontier to analyze the bit-equivalent
definition of open-ended learning proposed by Xu, Zhu, and Van Roy (2026).
All information quantities are in nats.

## Materials

- [`manuscript.md`](manuscript.md): paper-level statement and narrative.
- [`../../docs/bit-equivalent-foundations.md`](../../docs/bit-equivalent-foundations.md):
  detailed theorem and proof draft.
- [`theorem-ledger.md`](theorem-ledger.md): assumption and claim ledger.
- [`assumption-stress-tests.md`](assumption-stress-tests.md): boundary and
  assumption-removal examples.
- [`systematic-literature-audit.md`](systematic-literature-audit.md): closest
  located antecedents and novelty risks.

## Requested review

### 1. Reward-transformation results

Please verify both directions of the universal maximality theorem:

- a positive-affine transformation conjugates every finite frontier;
- the one-action-lottery argument forces any continuous increasing universal
  transformation to be affine when the threshold relabeling is an order
  isomorphism onto the transformed reward interval.

For the cubic classification reversal, please check:

- the Donsker--Varadhan moment bound uniformly over the countable action set;
- the induced marginal action channel of the alternating query/deploy policy;
- the \(\Omega(T)\) average-bit-equivalent calculation;
- the activated oracle channel's exact mutual information;
- the distinction between transforming deterministic feedback and
  transforming a noisy conditional mean;
- whether the finite-per-round-mean policy convention is sufficient and
  natural for the claimed non-open-endedness conclusion.

### 2. Representation results

Please check the source-reduction proof and the action-quotient theorem in the
finite setting. In particular, assess whether the proposed standard-Borel
extension needs assumptions beyond regular conditional distributions,
measurable source-independent lifts, integrable reward, and closure of the
admissible channel class.

Please also assess whether “reward-sufficient” is established terminology here
or whether the result should instead be framed through Bayesian sufficiency,
remote rate--distortion, or Blackwell comparison.

### 3. Nondegeneracy and collapse

Please verify:

- the Pinsker constant under the stated total-variation and natural-log
  conventions;
- the distinction between \(B(\rho)=0\) attained by an independent channel and
  a nonattained zero infimum;
- the uniform-integrability argument for the varying law pairs
  \((P_n,P_\Theta P_{A,n})\);
- the compact-continuous attainment theorem;
- the strict-margin and bounded-noncompact counterexamples.

The draft claims only that loss of uniform integrability is **necessary** for
vanishing-information sequences at a fixed positive gap above \(R_0\). Please
flag any sentence that accidentally implies sufficiency.

### 4. Composition

Finite independent infimal convolution is treated as classical. The main
technical review target is the countable iid theorem under the following
contract:

- finite component source and action alphabets;
- a pointwise zero-reward null action;
- independent sources, product actions, and additive reward;
- actions with finite support relative to the null action;
- finite expected support size.

Please scrutinize the finite-coordinate mutual-information lower bound, the
regular-conditional reduction to component channels, the Fubini passage, and
the approximately optimal product-channel limit establishing

\[
B_\infty(\rho)
=
\rho\lim_{x\downarrow0}\frac{B_1(x)}x.
\]

We especially want either a counterexample within this contract or a precise
statement of why none exists.

### 5. Novelty and positioning

Please distinguish theorem novelty from novelty of application. The draft
currently treats the following as classical or elementary:

- the static utility-constrained mutual-information frontier;
- positive-affine expected-utility invariance;
- finite-alphabet convexity and attainment;
- data processing, Blackwell monotonicity, and finite tensorization;
- Pinsker and Bernoulli time-sharing arguments.

The intended new contribution is the foundations synthesis for the 2026
open-endedness definition, especially the invertible monotone classification
reversal, the four-way attained-zero/boundary-nonattainment/positive-gap-
collapse/infeasible taxonomy, strict-margin collapse examples, and the
countable local-price interpretation. Please identify any
prior theorem or counterexample that materially narrows that claim.

## Requested response format

For each main result, please return:

1. `correct`, `correct with revisions`, or `incorrect`;
2. the weakest assumptions under which it is valid;
3. any missing measurability, integrability, or extended-real convention;
4. the closest prior result you know;
5. whether the manuscript's novelty language is defensible.

Priority should be given to mathematical counterexamples and scope errors over
stylistic suggestions.
