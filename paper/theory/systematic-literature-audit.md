# Systematic literature and novelty audit

**Audit date:** 2026-08-01

**Scope:** the four-result theory package in
`docs/bit-equivalent-foundations.md` and `paper/theory/manuscript.md`

**Status:** conservative working audit, not a legal or bibliometric guarantee of
novelty

## Executive conclusion

The paper has a defensible contribution, but the novelty is narrower than “a new
reward--information frontier.” The static optimization

\[
\inf_{P(A\mid\Theta):\mathbb E U(\Theta,A)\geq \rho} I(\Theta;A)
\]

is ordinary rate--distortion theory with negative distortion and was explicitly
presented as a **rate-utility function** for decision making by Genewein et al.
(2015). Rational-inattention and information-theoretic bounded-rationality work
also optimizes the same mutual-information/expected-utility tradeoff. Xu, Zhu,
and Van Roy's 2026 contribution is the *bit-equivalent name* and its use inside
their definition of open-ended learning, not the underlying static program.

The safest paper-level claim is therefore:

> We analyze the semantic robustness of the recently proposed
> bit-equivalent/open-endedness criterion by applying and extending classical
> rate--distortion, expected-utility, and statistical-experiment theory. We give
> explicit counterexamples showing that invertible monotone reward changes and
> uncontrolled reward tails can reverse or collapse the resulting
> open-endedness classification, and we state exact representation and
> composition laws under explicit admissibility conditions.

The strongest apparently new item found in this audit is the **explicit
invertible monotone transformation that reverses the Xu--Zhu--Van Roy
open-endedness classification while preserving the feedback experiment**. The
countable local-price law may be a new corollary under the paper's particular
finite-expected-support contract, but its mathematical core is elementary
convex allocation and its literature risk remains high. The representation,
boundedness, finite-composition, compactness, and affine-invariance results are
classical or direct specializations and should be labeled as such.

No exact prior statement of the cubic classification-reversal construction was
located. That is evidence for a focused novelty claim, not proof of priority.

## Method and limitations

This was a primary-source-first search. I inspected:

- the full HTML of Xu, Zhu, and Van Roy, arXiv:2606.08369v1;
- the full text of Shannon's 1959 fidelity-criterion paper, including its
  sections on the zero-rate cutoff and product sources;
- publisher or proceedings pages from IEEE, Project Euclid, AEA, PMLR,
  Frontiers, NBER, and arXiv;
- DOI and bibliographic metadata for older sources when publisher full text was
  unavailable.

Search-engine coverage of a June 2026 proposal is incomplete, and older
rate--distortion results are often buried in monographs rather than titled with
modern terms such as “tensorization.” This audit did not include subscription
searches in MathSciNet, zbMATH, Web of Science, Scopus, or a complete
citation-network traversal. An information theorist should still check Berger,
Csiszár--Körner, Gray, and Han at theorem level before submission.

## 1. The frontier and adjacent decision theories

### Closest antecedents

1. [Shannon, “Coding Theorems for a Discrete Source With a Fidelity
   Criterion” (1959)](https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf)
   defines the minimum mutual information subject to an expected-distortion
   constraint. Setting distortion equal to a constant minus reward gives the
   present finite frontier immediately.
2. [Genewein, Leibfried, Grau-Moya, and Braun, “Bounded Rationality,
   Abstraction, and Hierarchical Decision-Making”
   (2015)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2015.00027/full)
   explicitly describes the dual problem as minimizing mutual information
   between world states and actions subject to a lower bound on expected
   utility, calls the resulting curve a *rate-utility function*, and interprets
   it as the minimum processing capacity required for a utility target.
3. [Sims, “Implications of Rational Inattention”
   (2003)](https://doi.org/10.1016/S0304-3932(03)00029-1),
   [Matějka and McKay, “Rational Inattention to Discrete Choices”
   (2015)](https://www.aeaweb.org/articles?id=10.1257/aer.20130047), and
   [Ortega and Braun, “Information, Utility & Bounded Rationality”
   (2011)](https://arxiv.org/abs/1107.5766) establish a broad literature in
   which decision rules trade expected utility against Shannon information.
4. [Xu, Zhu, and Van Roy, “An Information-Theoretic Definition for Open-Ended
   Learning” (2026)](https://arxiv.org/abs/2606.08369) defines the
   bit-equivalent and average bit-equivalent and makes their growth an
   open-endedness criterion. Version 1 does not discuss affine or nonlinear
   reward transformations, Blackwell equivalence, uniform integrability, or
   tensorization; full-text searches for those terms returned no matches.

### Consequence for positioning

- **Overlap:** the mathematical object (B(\rho)), finite-alphabet convexity,
  Lagrange duality, and its interpretation as an information price for utility
  are established prior art.
- **Safe wording:** “We study the bit-equivalent as the rate-utility frontier
  underlying Xu et al.'s open-endedness proposal.”
- **Do not write:** “We introduce a new Bayesian reward--information
  frontier,” or imply that interpreting mutual information as the processing
  required for an expected-utility target is new.
- **Residual risk:** low on this conclusion; Genewein et al. state the same
  static optimization and interpretation explicitly.

## 2. Finding I: reward transformations

### Claim I.a: positive-affine frontier conjugacy

**Closest antecedent.** Expected utility is cardinal: its lottery
representation is unique only up to positive-affine transformations. See
[von Neumann and Morgenstern, *Theory of Games and Economic Behavior*
(1944)](https://www.jstor.org/stable/j.ctt1r2gkx) and the mixture-space
axiomatization in [Herstein and Milnor, “An Axiomatic Approach to Measurable
Utility” (1953)](https://doi.org/10.2307/1905540). The same invariance is
routine in reinforcement learning; [Ng, Harada, and Russell, “Policy
Invariance Under Reward Transformations”
(1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
starts from positive-linear/affine utility invariance and studies the additional
potential-shaping freedom of MDPs.

- **Overlap:** (r' = \alpha r+\beta) simply relabels the expected-reward
  constraint; the theorem is a one-line rate--distortion/expected-utility
  identity.
- **Safe novelty wording:** “We record the exact conjugacy needed to transport
  the bit-equivalent and, under a bijective reward-history transformation, its
  sequential criterion.”
- **Novelty assessment:** classical specialization; do not advertise as an
  information-theory theorem.
- **Unresolved risk:** low. The sequential history-transport statement needs
  its stated observation assumptions, but is also routine.

### Claim I.b: positive-affine maps are the maximal universal pointwise class

**Closest antecedent.** The von Neumann--Morgenstern/Herstein--Milnor uniqueness
theorem already says that preserving expected-lottery comparisons universally
forces positive affinity. The manuscript's one-action-lottery/Jensen proof is
essentially that classical argument with frontier cutoffs used to recover the
same condition.

- **Overlap:** nearly all of the mathematical content is expected-utility
  uniqueness and Jensen equality.
- **Safe novelty wording:** “We give a direct frontier-level formulation of
  classical positive-affine uniqueness.” If desired: “Within the
  bit-equivalent formalism, this identifies the maximal universal pointwise
  invariance class.”
- **Do not write:** “We discover that only affine utility transformations
  preserve expected decisions.”
- **Unresolved risk:** medium. The exact universal-conjugacy formulation may
  not have appeared verbatim, but it is an immediate corollary of old theory;
  a priority claim would add little value and would be hard to defend.

### Claim I.c: an invertible monotone transformation reverses open-endedness

**Closest antecedents.** Expected-utility theory already explains why a
nonlinear increasing transformation changes lottery rankings. The vanishing
duty-cycle/increasing-payoff mechanism is analogous to on--off or flash
signaling: [Verdú, “On Channel Capacity per Unit Cost”
(1990)](https://doi.org/10.1109/18.57201) treats zero-cost symbols and
information per unit cost, while [Verdú, “Spectral Efficiency in the Wideband
Regime” (2002)](https://web.mit.edu/6.441/www/reading/verdu_wideband.pdf)
describes capacity-achieving on--off signaling with vanishing duty cycle and,
in relevant channels, unbounded amplitude. These are analogies, not the same
theorem.

- **Overlap:** non-affine transforms changing expected-lottery comparisons is
  classical; rare activation with growing amplitude is a classical
  information-theoretic mechanism.
- **Apparently distinct content:** the present construction couples an
  infinite-bit decision problem, the Xu et al. average-bit-equivalent growth
  definition, a Donsker--Varadhan lower bound before transformation, and a
  cubic time-sharing collapse after transformation, while the deterministic
  reward feedback remains invertibly recoverable.
- **Safe novelty wording:** “We give an explicit counterexample in which an
  invertible monotone pointwise reward transformation reverses the
  Xu--Zhu--Van Roy open-endedness classification despite Blackwell-equivalent
  feedback.” Prefer “we give” over “the first.”
- **Unresolved risk:** medium. No exact antecedent was located using searches
  combining *open-endedness*, *bit-equivalent*, *monotone reward
  transformation*, *rate-utility*, and *flash signaling*. Broader reward-hacking,
  risk-sensitive control, and utility-rescaling literatures could contain
  conceptually similar examples.

## 3. Finding II: source and action representation

### Claim II.a: reward-sufficient source reduction

**Closest antecedents.** Blackwell's comparison of experiments makes a signal
ordering equivalent to decision performance across loss functions; see
[Blackwell, “Equivalent Comparisons of Experiments”
(1953)](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.full).
Indirect/remote rate--distortion theory reduces a noisy-observation problem by
conditional expectation; see [Dobrushin and Tsybakov, “Information
Transmission With Additional Noise”
(1962)](https://doi.org/10.1109/TIT.1962.1057738) and
[Witsenhausen, “Indirect Rate Distortion Problems”
(1980)](https://doi.org/10.1109/TIT.1980.1056251). Decision-focused compression
is also central to [Arumugam and Van Roy, “Deciding What to Learn”
(2021)](https://proceedings.mlr.press/v139/arumugam21a.html).

- **Overlap:** conditioning/averaging within fibers and applying data
  processing are standard sufficiency arguments. If reward factors exactly
  through (S=s(\Theta)), optimizing over (\Theta\)-channels cannot beat the
  averaged (S\)-channel.
- **Safe novelty wording:** “We specialize sufficiency/data processing to
  identify the payoff-relevant source representation of the bit-equivalent.”
- **Novelty assessment:** classical lemma with useful semantic consequences;
  not a new sufficiency theorem.
- **Unresolved risk:** low for the finite result. A standard-Borel extension
  should cite regular-conditional-probability results and not be claimed until
  its hypotheses are explicit.

### Claim II.b: behavioral action quotient and admissible lift

**Closest antecedents.** In rate--distortion theory, the reproduction alphabet
is defined only through the distortion columns (d(x,\hat x)); duplicate
columns are redundant. In decision theory, acts with identical statewise
payoffs are behaviorally equivalent. The two data-processing directions in the
manuscript are elementary once a source-independent measurable lift exists.
Value-preserving abstractions are also studied in RL, for example
[Abel et al., “Value Preserving State-Action Abstractions”
(2020)](https://proceedings.mlr.press/v108/abel20a.html), although that paper's
dynamic setting and goal differ.

- **Overlap:** quotient pushforward cannot increase information; a stochastic
  section recovers equality. This is standard channel/data-processing logic.
- **Safe novelty wording:** “We state the exact lifting condition under which
  raw action serialization is immaterial to the optimized bit-equivalent.”
- **Do not write:** “We introduce behavioral action quotienting.”
- **Unresolved risk:** medium for general measurable spaces and constrained
  admissibility. The finite theorem is routine; measurable selection and
  resource preservation are where a nontrivial extension would lie.

### Claim II.c: Blackwell monotonicity

- **Closest antecedent:** Blackwell (1951/1953), directly.
- **Overlap:** complete. Simulating a garbling before acting reproduces any
  decision rule available under the less informative experiment.
- **Safe wording:** label this a proposition or consistency check, not a
  contribution. Make clear that it concerns an experiment-restricted frontier;
  the unrestricted static frontier corresponds to observing the source.
- **Unresolved risk:** none on novelty; it is classical.

## 4. Finding III: nondegeneracy, nonattainment, and tail escape

### Claim III.a: the zero-information baseline

**Closest antecedent.** Shannon's 1959 paper proves for finite alphabets that
rate zero occurs exactly when source and reproduction are independent and that
the smallest zero-rate distortion is achieved by the best constant
reproduction symbol. Under (d=C-r), this is exactly the manuscript's best
source-independent reward (R_0) and its finite attained-zero lemma.

- **Overlap:** complete in the finite case.
- **Safe wording:** “The classical zero-rate cutoff becomes the uninformed
  reward baseline (R_0). We distinguish attainment from a zero-valued
  infimum on noncompact spaces.”
- **Unresolved risk:** low for finite alphabets; medium at the noncompact
  boundary because existence theory is extensive.

### Claim III.b: bounded-reward Pinsker certificate

**Closest antecedents.** This is the standard bound on the expectation of a
bounded function by total variation followed by the
Csiszár--Kullback--Pinsker inequality; historical sources include
[Pinsker, *Information and Information Stability of Random Variables and
Processes* (1964)](https://openlibrary.org/books/OL5912677M/Information_and_information_stability_of_random_variables_and_processes)
and [Csiszár, “Information-Type Measures of Difference of Probability
Distributions and Indirect Observations”
(1967)](https://ndlsearch.ndl.go.jp/en/books/R100000136-I1572824501190134016).

- **Overlap:** the proof and constant are a direct classical application.
- **Safe novelty wording:** “Pinsker's inequality supplies an explicit
  positive-gap certificate for the bit-equivalent.”
- **Novelty assessment:** not new; useful as a clean robustness corollary.
- **Unresolved risk:** low, provided the total-variation convention and natural
  logarithms remain explicit.

### Claim III.c: uniform integrability is necessary for positive-gap collapse

**Closest antecedents.** The analytic step—total-variation convergence plus
uniform integrability implies convergence of expectations—is a standard
Vitali/truncation argument. A modern primary treatment for varying measures is
[Feinberg, Kasyanov, and Liang, “Fatou's Lemma for Weakly Converging Measures
Under the Uniform Integrability Condition”
(2018)](https://arxiv.org/abs/1807.07931). Uniform-integrability conditions
also occur throughout general-source rate--distortion theory; this means the
tail condition itself should not be marketed as imported into information
theory for the first time.

- **Overlap:** the theorem follows immediately after Pinsker converts
  (I(P_n\Vert Q_n)\to0) to total-variation convergence and reward is truncated.
- **Potentially useful synthesis:** applying that standard lemma to the
  dependent law (P_n) and its product-of-marginals reference (Q_n) yields a
  crisp necessary condition for bit-equivalent collapse above (R_0).
- **Safe novelty wording:** “We derive a tail-escape diagnostic: any
  vanishing-information sequence maintaining a fixed reward gap above (R_0)
  must violate the stated two-law uniform-integrability condition.”
- **Do not write:** “We characterize collapse.” The result is necessary, not
  sufficient, and it requires UI under both (P_n) and (Q_n).
- **Unresolved risk:** medium. A theorem with essentially this exact
  product-reference formulation may exist in robust statistics,
  entropy-constrained optimization, or general-alphabet rate--distortion.

### Claim III.d: rare-burst and high-leverage collapse examples

**Closest antecedents.** Bernoulli time sharing is elementary. The closest
information-theoretic mechanism is Verdú's capacity-per-unit-cost and flash
signaling work cited above: a zero-cost/idle symbol is used most of the time,
while rare nonzero symbols carry disproportionate value. The direction of
optimization here is different—minimum source information for fixed expected
reward rather than maximum communicated information per input cost.

- **Overlap:** vanishing duty cycle and increasing amplitude are classical.
- **Apparently distinct content:** the rare-burst lemma, strict-margin
  counterexample, and small-correlation/high-leverage example directly expose
  nonattained zero information in the bit-equivalent definition.
- **Safe novelty wording:** “We exhibit two concrete collapse mechanisms and
  relate them to flash signaling.”
- **Unresolved risk:** medium-high. Search terms such as *minimum mutual
  information utility constraint rare event*, *rate-utility unbounded payoff*,
  and *rational inattention unbounded utility nonattainment* did not locate the
  same examples, but economics may contain analogous unbounded-utility
  pathologies under different terminology.

### Claim III.e: compact bounded-continuous attainment

**Closest antecedents.** Existence of rate--distortion optimizers on compact
alphabets follows from weak compactness and lower semicontinuity and is
classical. The difficulty on noncompact reproduction spaces remains active;
for context, [Zou et al., “Rate-Distortion Theory on Non-Compact Spaces”
(2026)](https://arxiv.org/abs/2601.07246) explicitly positions compactness as
the classical existence route and develops coercive substitutes.

- **Overlap:** the manuscript's proof is a standard direct-method argument.
- **Safe wording:** “Under a compact-continuous contract, standard existence
  theory closes the nonattainment loophole.”
- **Novelty assessment:** not new.
- **Unresolved risk:** low for the stated compact theorem; high if extended to
  noncompact spaces without engaging modern existence literature.

## 5. Finding IV: finite and countable composition

### Claim IV.a: finite independent-product infimal convolution

**Closest antecedent.** Shannon's 1959 paper contains a section titled “Rate
for a Product Source with a Sum Distortion Measure.” For two independent
sources it proves that a product test channel is optimal and gives

\[
R(d)=\min_t\{R_1(t)+R_2(d-t)\},
\]

with the optimal allocation characterized by equal slopes. Iteration gives the
finite-(n) result. With (d_i=C_i-r_i), this is the manuscript's finite
infimal-convolution theorem.

- **Overlap:** complete. Even the entropy-chain converse and product-channel
  achievability are Shannon's argument.
- **Safe wording:** “Translating Shannon's product-source theorem into reward
  coordinates gives the finite composition identity.”
- **Do not list as a novel contribution.** It can remain as a foundational
  proposition needed for the countable analysis.
- **Unresolved risk:** none on novelty; this is explicit primary prior art.

### Claim IV.b: identical finite tensorization

- **Closest antecedent:** Shannon's product theorem plus convexity/equal-slope
  allocation.
- **Overlap:** complete.
- **Safe wording:** “The classical identical-coordinate specialization is
  (B_n(\rho)=nB_1(\rho/n)).”
- **Unresolved risk:** none on novelty.

### Claim IV.c: countable local-price law

**Closest antecedents.** The finite part is Shannon. The passage to a
countable family resembles infinite resource allocation and countable infimal
convolution in convex analysis; [Rockafellar, *Convex Analysis*
(1970)](https://press.princeton.edu/books/paperback/9780691015866/convex-analysis)
is the classical background. Gaussian reverse-waterfilling and general-source
rate--distortion theory also allocate distortion across indefinitely many
spectral components, although no source located in this audit stated the
manuscript's exact finite-expected-support theorem.

- **Overlap:** once finite composition is known, the upper bound
  (nB_1(\rho/n)\to\kappa\rho) and the lower bound
  (B_1(x)\geq\kappa x) are elementary convex analysis. The information-theory
  work lies in justifying finite-coordinate projection and the exchange of the
  countable reward sum under the admissibility contract.
- **Safe novelty wording:** “Under an explicit pointwise-null and
  finite-expected-support contract, we derive a countable-product corollary in
  which the frontier linearizes at the component frontier's right slope at
  zero.”
- **Novelty assessment:** possibly new as a precisely delimited corollary for
  the bit-equivalent, but probably not a deep new tensorization principle.
- **Unresolved risk:** high. Search queries using *countable product
  rate-distortion*, *independent nonidentically distributed sources*, *infinite
  infimal convolution*, *local slope at zero*, *parallel sources*, and
  *resource allocation* found related waterfilling and product-source work but
  no exact match. Older books and nonstationary/general-source coding results
  require expert review. The theorem must retain its null action,
  finite-expected-support, bounded component reward, iid source, and additive
  reward assumptions.

## 6. Claim-by-claim novelty ledger

| Current claim | Closest antecedent | Defensible status | Literature risk |
|---|---|---|---|
| Static bit-equivalent frontier | Shannon 1959; Genewein et al. 2015; rational inattention | Established rate-utility object; new use/name comes from Xu et al. | Low |
| Positive-affine conjugacy | Expected-utility uniqueness; ordinary rate--distortion | Classical specialization | Low |
| Maximal universal pointwise class | vNM/Herstein--Milnor plus Jensen equality | Direct frontier-level reformulation | Medium |
| Cubic open-endedness reversal | Cardinal utility; flash signaling analogy | Strongest apparently new counterexample | Medium |
| Reward-sufficient source reduction | Sufficiency, data processing, indirect RDF | Classical semantic lemma | Low |
| Behavioral action quotient | Duplicate distortion columns; data processing | Elementary lemma; lifting condition is useful | Medium on measurable extensions |
| Blackwell monotonicity | Blackwell 1951/1953 | Classical consistency result | None |
| Attained zero-information cutoff | Shannon's (R(D)=0) cutoff | Classical in finite alphabets | Low |
| Bounded Pinsker lower bound | Pinsker/Csiszár | Classical corollary | Low |
| UI necessity for positive-gap collapse | Pinsker plus Vitali/truncation | Useful application/synthesis, not a new analytic theorem | Medium |
| Rare-burst/high-leverage examples | Bernoulli time sharing; Verdú flash signaling | Apparently distinct bit-equivalent pathologies | Medium-high |
| Compact attainment | Classical RDF existence by compactness/l.s.c. | Classical | Low |
| Finite infimal convolution | Shannon 1959 product-source theorem | Explicit prior art | None |
| Identical finite tensorization | Shannon plus convexity | Explicit prior art/corollary | None |
| Countable local-price law | Shannon plus convex analysis/infinite allocation | Possibly new delimited corollary | High |

## 7. Recommended manuscript language

The abstract and introduction should distinguish three layers:

1. **Established object:** the bit-equivalent is a rate-utility/rate--distortion
   frontier already familiar in information-constrained decision making.
2. **New target of analysis:** Xu et al. use that frontier to define
   open-endedness, creating new semantic requirements not addressed by the
   classical coding interpretation.
3. **Paper contribution:** an explicit classification reversal and collapse
   examples, supported by a careful synthesis of classical invariance,
   sufficiency, nondegeneracy, and composition results.

Suggested contribution verbs:

- “record,” “specialize,” or “translate” for affine conjugacy, Pinsker,
  compactness, Blackwell monotonicity, and finite composition;
- “formalize” for source reduction and behavioral quotients;
- “derive under an explicit contract” for the countable local-price law;
- “construct” or “exhibit” for the monotone classification reversal and
  collapse examples.

Avoid “first,” “novel theorem,” and “new information-theoretic frontier” until
an expert has completed citation chaining. The paper can still make a strong
theoretical contribution: exposing a robustness failure of a new definition is
valuable even when the diagnostic tools are classical.

## 8. Reproducible search log

Searches were run on **2026-08-01**. The following are representative exact
queries; publisher/domain variants and title searches were also used.

### Frontier, inverse rate--distortion, and rational inattention

- `"minimum mutual information" "expected utility" decision`
- `"rate-utility function" information theory mutual information`
- `"information constrained" expected utility mutual information rate distortion`
- `site:frontiersin.org "Rate-Utility" function bounded rationality`
- `site:aeaweb.org rational inattention discrete choices mutual information`
- `site:arxiv.org information utility bounded rationality Ortega Braun`

### Reward invariance

- `von Neumann Morgenstern expected utility unique up to positive affine transformation`
- `Herstein Milnor 1953 axiomatic approach measurable utility original paper`
- `reward transformations preserve optimal policies positive affine MDP theorem`
- `"maximal universal invariance" "rate-utility"`
- `"monotone" "open-endedness" reward transformation information`
- `open-ended learning reward transformation invariance monotone affine`

### Sufficiency, Blackwell, and remote rate--distortion

- `Blackwell 1953 equivalent comparisons experiments Project Euclid`
- `Witsenhausen indirect rate distortion problems original paper`
- `Dobrushin Tsybakov information transmission with additional noise`
- `rate distortion identical reproduction symbols same distortion columns`
- `decision theory redundant actions same utility all states remove action`
- `Arumugam Van Roy Deciding What to Learn rate distortion PMLR`

### Zero rate, nonattainment, UI, and flash mechanisms

- `rate distortion zero rate threshold Dmax constant reproduction symbol`
- `rate distortion nonattainment general alphabets compactness lower semicontinuity`
- `Csiszar 1974 On an extremum problem of information theory rate distortion`
- `"uniform integrability" "rate-distortion"`
- `total variation convergence uniform integrability convergence expectations`
- `minimum mutual information utility constraint rare event`
- `rational inattention unbounded utility nonattainment`
- `Verdu channel capacity per unit cost zero cost symbol`
- `flash signaling vanishing duty cycle increasing amplitude mutual information`

### Product and countable composition

- `rate distortion independent sources infimal convolution Shannon 1959`
- `rate distortion product source additive distortion tensorization theorem`
- `rate distortion independent nonidentically distributed sources allocation`
- `rate distortion infinite product countably many independent components`
- `"local slope" "rate-distortion" countable product`
- `countable infimal convolution identical convex function local slope at zero`
- `resource allocation countably many identical convex costs local slope at origin`

### Negative-result interpretation

No exact match was found for the cubic open-endedness reversal or for the
countable finite-expected-support local-price theorem. Search failure is most
informative for the former because “bit-equivalent” and the Xu et al.
open-endedness definition are new, sharply searchable terms. It is less
informative for the latter because old product-source and convex-allocation
results use many different vocabularies.

## 9. Primary-source shortlist for the paper

- C. E. Shannon (1959), [“Coding Theorems for a Discrete Source With a
  Fidelity Criterion”](https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf).
- T. Genewein, F. Leibfried, J. Grau-Moya, and D. A. Braun (2015),
  [“Bounded Rationality, Abstraction, and Hierarchical
  Decision-Making”](https://doi.org/10.3389/frobt.2015.00027).
- C. A. Sims (2003), [“Implications of Rational
  Inattention”](https://doi.org/10.1016/S0304-3932(03)00029-1).
- F. Matějka and A. McKay (2015), [“Rational Inattention to Discrete
  Choices”](https://doi.org/10.1257/aer.20130047).
- J. von Neumann and O. Morgenstern (1944), [*Theory of Games and Economic
  Behavior*](https://www.jstor.org/stable/j.ctt1r2gkx).
- I. N. Herstein and J. Milnor (1953), [“An Axiomatic Approach to Measurable
  Utility”](https://doi.org/10.2307/1905540).
- D. Blackwell (1953), [“Equivalent Comparisons of
  Experiments”](https://doi.org/10.1214/aoms/1177729032).
- R. L. Dobrushin and B. S. Tsybakov (1962), [“Information Transmission With
  Additional Noise”](https://doi.org/10.1109/TIT.1962.1057738).
- H. S. Witsenhausen (1980), [“Indirect Rate Distortion
  Problems”](https://doi.org/10.1109/TIT.1980.1056251).
- D. Arumugam and B. Van Roy (2021), [“Deciding What to Learn: A
  Rate-Distortion Approach”](https://proceedings.mlr.press/v139/arumugam21a.html).
- M. S. Pinsker (1964), [*Information and Information Stability of Random
  Variables and Processes*](https://openlibrary.org/books/OL5912677M/Information_and_information_stability_of_random_variables_and_processes).
- I. Csiszár (1967), [“Information-Type Measures of Difference of Probability
  Distributions and Indirect
  Observations”](https://ndlsearch.ndl.go.jp/en/books/R100000136-I1572824501190134016).
- S. Verdú (1990), [“On Channel Capacity per Unit
  Cost”](https://doi.org/10.1109/18.57201).
- S. Verdú (2002), [“Spectral Efficiency in the Wideband
  Regime”](https://doi.org/10.1109/TIT.2002.1003824).
- W. Xu, Y. Zhu, and B. Van Roy (2026), [“An Information-Theoretic Definition
  for Open-Ended Learning”](https://arxiv.org/abs/2606.08369).
