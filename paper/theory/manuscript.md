# When Does Reward Require Information?
## Invariances and Degeneracies of the Bit-Equivalent

**Status:** Theory manuscript draft. All theorem numbering and novelty claims
are provisional pending full proofs and literature review.

## Abstract

The bit-equivalent of a reward level is the infimum mutual information between
an unknown environment and an action required to attain that reward in
expectation. It has recently been proposed as the basis of an
information-theoretic definition of open-ended learning. We study the
foundations of this quantity as a Bayesian reward--information frontier. First,
we prove that positive-affine reward transformations preserve the frontier and
average bit-equivalent exactly, and that they are the only pointwise
transformations with this property uniformly over finite decision problems.
We then construct an environment whose open-endedness classification is
reversed by an invertible monotone reward transformation despite unchanged
feedback information. Second, for finite Bayesian decision problems we prove
invariance under reward-sufficient reductions of the environment and
behavior-preserving quotients of the action space that admit a
source-independent admissible lift. Third, we give a
quantitative lower bound preventing zero-information collapse strictly above
the best source-independent reward when reward has bounded oscillation, and we
show that any vanishing-information sequence maintaining a fixed positive
reward gap must lose uniform integrability. Finally, we derive an
infimal-convolution law for finite independent additive decision problems,
recover identical finite tensorization, and prove a countable-product
local-slope formula under an explicit finite-expected-support contract.
Together, these
results clarify when reward-relevant information is invariant, when it is
nondegenerate, and when local decision demands compose into an open-ended
frontier. They also show that the resulting notion of open-endedness is
cardinal rather than ordinal and requires explicit control of reward tails.

## 1. Introduction

Open-ended learning is intended to describe systems whose useful capabilities
can continue to grow. Reward, novelty, task count, and total information each
capture only part of that idea. A recently proposed alternative is the
bit-equivalent

\[
B_\rho
=
\inf_{P(A\mid\Theta):\mathbb E[r_\Theta(A)]\geq\rho}
I(\Theta;A),
\]

which prices a reward target by the least information that successful behavior
must contain about the environment. Xu, Zhu, and Van Roy (2026) then call an
environment open-ended when some interacting agent sustains linear growth in
the average bit-equivalent of its rewards.

This definition raises foundational questions before questions of measurement
or algorithm design. Does the quantity depend on irrelevant coordinates in
the declared environment parameter? Can redundant serializations alter the
optimized frontier of the same behavior? Which changes of reward units
preserve the classification? Can expected-reward lotteries drive the infimum
to zero even when every successful deterministic behavior requires
information? Finally, when does combining many individually bounded decision
problems create an unbounded reward--information frontier?

We address these questions directly. The resulting picture has two sides. The
frontier has strong semantic invariances: it can be optimized over a
reward-sufficient state and over behavioral action quotients with admissible
lifts, and it is exactly invariant to positive-affine reward changes. An
individual raw channel may still encode avoidable serialization information.
The frontier also has genuine degeneracies. General
monotone reward transformations do not preserve expected-reward comparisons,
and unbounded rare rewards can make positive performance attainable at
arbitrarily small information cost. These are properties of the mathematical
definition rather than failures of a numerical estimator.

### 1.1 Contributions

Our target contributions are:

1. **Reward relativity.** We prove exact positive-affine invariance and a
   universal maximality theorem. We give an explicit infinite decision problem
   that is open-ended under reward \(r\) and non-open-ended under the
   invertible transformation \(r^3\), while its feedback remains recoverable.
2. **Representation invariance.** We prove exact reduction to
   reward-sufficient environment statistics and exact quotienting of
   behaviorally equivalent actions whenever a source-independent admissible
   lift exists.
3. **Nondegeneracy.** We lower-bound the frontier above the best uninformed
   reward whenever reward has bounded oscillation. We show that any
   vanishing-information sequence maintaining reward a fixed positive amount
   above that baseline must lose uniform integrability.
4. **Composition.** We prove an infimal-convolution identity for finite
   independent additive decision problems and recover identical-coordinate
   finite tensorization. Under a pointwise null action and a
   finite-expected-support contract, we prove that the countable frontier is
   linear with slope equal to the component frontier's local slope at zero.

The elementary ingredients—rate--distortion convexity, data processing,
Pinsker's inequality, and expected-utility affine invariance—are classical.
Our aim is to determine their joint consequences for the new definition and
to isolate the additional conditions required for an intrinsic interpretation
of open-ended capability.

## 2. Bayesian reward--information frontiers

Let

\[
\mathsf D=(\mathcal T,\mu,\mathcal A,r)
\]

be a finite Bayesian decision problem. A behavioral channel
\(K(a\mid\theta)\) induces a joint law \(P_{\Theta,A}=\mu K\), expected reward
\(R(K)=\mathbb E[r(\Theta,A)]\), and information cost
\(I(K)=I(\Theta;A)\). We study the extended-real frontier

\[
B_{\mathsf D}(\rho)
=
\inf_{K:R(K)\geq\rho}I(K),
\]

where an empty feasible set has value \(+\infty\). Information is measured in
nats. The best reward available without source information is

\[
R_0(\mathsf D)
=
\sup_{\nu\in\mathcal P(\mathcal A)}
\mathbb E_{\mu\otimes\nu}[r(\Theta,A)]
=
\sup_a\mathbb E_\mu[r(\Theta,a)].
\]

The supremum is attained in the finite setting but not necessarily on a
noncompact action space. This distinction will matter: \(B(\rho)=0\) can mean
that an independent action actually attains \(\rho\), or only that a sequence
of increasingly extreme channels approaches zero information.

The finite frontier is nondecreasing and convex. Mathematically it is a
reward-sign version of Shannon's classical rate--distortion function. Our claims are
therefore not that this static optimization is new, but that its use as a
definition of open-endedness creates unresolved semantic and limiting
questions.

## 3. Reward semantics

The first question is which reward changes leave the frontier—and the proposed
open-endedness classification—unchanged.

### Theorem 1 (positive-affine invariance)

If \(r'=\alpha r+\beta\) with \(\alpha>0\), then

\[
B_{r'}(\rho)
=
B_r\!\left(\frac{\rho-\beta}{\alpha}\right).
\]

This is exact because the two reward constraints select the same behavioral
channels after relabeling the threshold. In a sequential environment whose
nonreward observations are unchanged and whose realized reward feedback is
also transformed affinely, policies transport through the bijection between
reward histories. The average bit-equivalent is then unchanged.

### Theorem 2 (maximal universal class)

Let \(J\) be an interval and \(g:J\to\mathbb R\) be continuous and strictly
increasing. If a continuous order isomorphism \(h:J\to g(J)\) conjugates the
frontier before and after applying \(g\) for every finite decision problem
with rewards in \(J\), then

\[
g(x)=\alpha x+\beta,
\qquad \alpha>0.
\]

The proof uses one-action lotteries. Constant lotteries identify the threshold
relabeling with \(g\); arbitrary two-point lotteries then force
\(\mathbb E[g(X)]=g(\mathbb E[X])\), hence Jensen equality and affinity. This
is the frontier analogue of the classical cardinal character of expected
utility.

The universal conclusion is sharp. A nonlinear map may be harmless on one
finite payoff support, but no larger pointwise class works for all problems.
Moreover, transforming a noisy realized reward is not the same operation as
transforming its conditional mean.

### Theorem 3 (invertible monotone classification reversal)

Let the hidden environment be an infinite sequence of independent fair bits.
An agent may query bit \(i\), receiving that bit as deterministic reward, or
deploy a proposed length-\(n\) prefix, receiving reward \(n\) exactly when it
is correct. A Donsker--Varadhan bound gives, for every
\(0<\lambda<\log2\),

\[
B^r(\rho)\geq\lambda\rho-\log2.
\]

An agent that alternates between learning the next bit and deploying the known
prefix therefore has average bit-equivalent \(\Omega(T)\).

Now replace every deterministic reward by its cube. For any finite
\(\rho>0\), consider the frontier test channel—not the learning agent—that
outputs the correct length-\(n\) prefix with probability \(d=\rho/n^3\) and
abstains otherwise. This channel has

\[
\mathbb E[r^3]=\rho,
\qquad
I(\Theta;A)=\frac{\rho\log2}{n^2}\longrightarrow0.
\]

Thus \(B^{r^3}(\rho)=0\) at every finite threshold, and every admissible policy
with finite per-round transformed mean reward has zero average bit-equivalent.
Cubing is bijective and the rewards are deterministic, so the original
feedback is exactly recovered by taking cube roots: the two feedback
experiments are Blackwell-equivalent. Nevertheless, the open-endedness
classification changes. The criterion is therefore cardinal, not ordinal.

## 4. Reward-sufficient states and behavioral actions

The second question is whether changing the representation of the source or
action changes the optimized information price.

### Theorem 4 (reward-sufficient source reduction)

Suppose \(S=s(\Theta)\) and reward factors as
\(r(\theta,a)=\bar r(s(\theta),a)\). Then

\[
B_\Theta(\rho)=B_S(\rho).
\]

A channel from \(S\) lifts through \(s\) without changing information.
Conversely, averaging any \(\Theta\)-channel within the fibers of \(S\)
preserves the joint law of \((S,A)\) and cannot increase mutual information.
This proves exact equality, not merely a data-processing bound.

Consequently, adding payoff-extraneous coordinates to the declared source does
not change the static optimized frontier, even when those coordinates are
correlated with the reward-sufficient state. Correlation can still make them
useful acquisition-time proxies, so this is not a statement about learning
trajectories or experimental distractors.

### Theorem 5 (behavioral action quotient)

Suppose \(q:\mathcal A\to\bar{\mathcal A}\) preserves reward:

\[
r(\theta,a)=\bar r(\theta,q(a)).
\]

Pushing actions through \(q\) yields

\[
B_{\bar{\mathcal A}}(\rho)\leq B_{\mathcal A}(\rho).
\]

Equality holds when the quotient admits a source-independent admissible lift
supported on each fiber. A raw channel may encode redundant serialization
information, but the optimized frontier can avoid that excess by using a
common lift. Outside finite spaces, measurability, feasibility, dynamics, and
resource preservation are part of this condition.

An experiment-restricted extension is monotone in Blackwell's order: a more
informative experiment can simulate a less informative one before applying
its decision rule. This is classical, but it locates the precise boundary
between source representation and information acquisition.

## 5. Nondegeneracy and rare rewards

The third question is when positive performance genuinely requires positive
information.

### Theorem 6 (bounded positive-gap certificate)

If reward lies in an interval of length \(L>0\), then for every
\(\rho>R_0\),

\[
B(\rho)
\geq
2\left(\frac{\rho-R_0}{L}\right)^2.
\]

For a joint law \(P=P_{\Theta,A}\) and its independence reference
\(Q=P_\Theta P_A\), bounded oscillation gives

\[
\mathbb E_P r-R_0
\leq L\lVert P-Q\rVert_{\mathrm{TV}},
\]

and Pinsker completes the proof. The strict gap is essential: a bounded
noncompact action family can have \(B(R_0)=0\) without attaining an
independent maximizer.

### Theorem 7 (tail escape is necessary for positive-gap collapse)

Let \(P_n\) be joint laws induced by channels with
\(I_{P_n}(\Theta;A)\to0\), and let \(Q_n=P_\Theta P_{A,n}\). If the reward
family is uniformly integrable under both \(P_n\) and \(Q_n\), then

\[
\mathbb E_{P_n}r-\mathbb E_{Q_n}r\to0.
\]

It follows that a vanishing-information sequence maintaining reward a fixed
positive amount above \(R_0\) must violate uniform integrability. Boundedness
is one sufficient tail contract, but not the only one.

This yields a useful three-way distinction:

1. **Attained zero information:** a source-independent action distribution
   actually reaches the threshold.
2. **Nonattained collapse:** every feasible channel has positive information,
   but a tail-escaping sequence drives the infimum to zero.
3. **Infeasibility:** no channel reaches the threshold.

The second case is easy to miss if an infimum is casually called a minimum.
It admits an on--off or flash-signaling-like construction: use increasingly
valuable informed behavior with a vanishing activation probability. Even a
strictly negative uninformed payoff for every nontrivial action does not stop
collapse when informed payoff magnitude is unbounded. Collapse can also arise
without rare activation, by combining vanishing source--action correlation
with increasing reward leverage.

Compactness closes the boundary loophole. With a Polish source, compact metric
action space, all Borel kernels, and bounded continuous reward, the frontier
attains every feasible value and \(R_0\) is attained. In this class,
\(B(\rho)=0\) exactly when \(\rho\leq R_0\).

## 6. Independent composition

The fourth question is how component information prices combine.

### Theorem 8 (finite infimal convolution)

For independent sources, product actions, and additive reward,

\[
B_{\otimes_i\mathsf D_i}(\rho)
=
\inf_{\sum_i\rho_i\geq\rho}
\sum_i B_{\mathsf D_i}(\rho_i).
\]

The converse follows from

\[
I(\Theta_{1:n};A_{1:n})
\geq\sum_i I(\Theta_i;A_i),
\]

which uses source independence. Product component channels attain the reverse
inequality. For identical components, convexity yields exact tensorization

\[
B_n(\rho)=nB_1(\rho/n).
\]

These finite results are classical rate--distortion structure and serve here
as a consistency requirement.

### Theorem 9 (countable local-price law)

Assume a finite component problem has a pointwise zero-reward null action.
In the countable iid product, restrict actions to finite support and require
finite expected support size. Let

\[
\kappa=\lim_{x\downarrow0}\frac{B_1(x)}x.
\]

Then for every finite \(\rho\geq0\),

\[
B_\infty(\rho)=\kappa\rho.
\]

Finite expected support and bounded component reward justify the infinite
reward sum. The lower bound follows by projecting to the first \(m\)
coordinates and then taking \(m\to\infty\). For achievability, distribute
\(\rho\) equally over the first \(n\) coordinates and let \(n\to\infty\).

Thus it is the local information price of an infinitesimal amount of reward,
not the number of available tasks by itself, that controls the countable
static frontier. If \(\kappa>0\), the frontier is linear; if \(\kappa=0\), it
collapses. This theorem does not construct a sequential learner.

## 7. Consequences for open-ended learning

The four answers suggest an explicit intrinsicness contract for using the
bit-equivalent as a capability price:

1. **Fix a cardinal reward scale.** Equivalence up to positive-affine changes
   is safe; arbitrary monotone changes are not.
2. **Declare only a payoff-sufficient source.** Extra latent coordinates do
   not belong in the semantic target, even if they affect acquisition.
3. **Price behavioral actions.** Raw representations should be quotiented only
   when a measurable, feasible, source-independent lift exists.
4. **Control tails and admissibility.** A bounded, uniformly integrable, or
   compact-continuous contract is needed to rule out unintended collapse.
5. **Separate frontier geometry from learnability.** Tensorization says what
   successful behavior must encode. It does not show that an interaction
   protocol lets an agent discover that behavior.

Under this contract, the frontier has a coherent interpretation. Without it,
two formally legal descriptions of essentially the same feedback process can
receive different open-endedness classifications, or an unbounded reward
lottery can make apparently demanding performance cost zero bits.

## 8. Discussion

The paper's contribution is a foundations layer, not a new estimator or
benchmark. Most individual tools—rate--distortion convexity, affine utility
invariance, data processing, Blackwell comparison, Pinsker's inequality, and
finite tensorization—are classical. The substantive result is their synthesis
for the new open-endedness definition, together with explicit counterexamples
that reverse or collapse its classification.

Several extensions remain. Prior sensitivity asks whether a capability price
should depend on one Bayesian source law. Sequential and stateful problems
require directed-information or causal analogues rather than a one-shot
channel. Statistical identification asks what can be inferred from black-box
interaction. Finally, a linear decision frontier does not provide a
schedule-free agent that attains it. These are distinct research programs; the
four foundational questions here determine whether the underlying target is
sound before those programs begin.

## References

- D. Arumugam and B. Van Roy, “Deciding What to Learn: A Rate-Distortion
  Approach,” 2021. <https://proceedings.mlr.press/v139/arumugam21a.html>
- D. Blackwell, “Equivalent Comparisons of Experiments,” 1953.
  <https://projecteuclid.org/euclid.aoms/1177729032>
- T. Genewein, F. Leibfried, J. Grau-Moya, and D. A. Braun, “Bounded
  Rationality, Abstraction, and Hierarchical Decision-Making,” 2015.
  <https://doi.org/10.3389/frobt.2015.00027>
- C. E. Shannon, “Coding Theorems for a Discrete Source With a Fidelity
  Criterion,” 1959.
  <https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf>
- S. Verdú, “On Channel Capacity per Unit Cost,” 1990.
  <https://doi.org/10.1109/18.57201>
- J. von Neumann and O. Morgenstern, *Theory of Games and Economic Behavior*,
  1944. <https://www.jstor.org/stable/j.ctt1r2gkx>
- W. Xu, Y. Zhu, and B. Van Roy, “An Information-Theoretic Definition for
  Open-Ended Learning,” 2026. <https://arxiv.org/abs/2606.08369>
