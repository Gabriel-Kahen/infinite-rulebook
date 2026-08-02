# Theorem ledger

## Purpose and audit verdict

This ledger separates mathematical validity, scope, and novelty for every
substantive result currently used by the manuscript. The authoritative detailed
statements and proofs are in `docs/bit-equivalent-foundations.md`; theorem
numbers in the manuscript differ after the reward-transformation section, so
both labels are recorded below.

**Audit verdict.** No contradiction, reversed inequality, or fatal proof gap
was found in the four main findings under the assumptions written in the
technical draft. The conclusions are substantially narrower than their
informal versions, however:

- frontier results are one-shot/static unless a row explicitly says otherwise;
- reward transformation acts on conditional mean reward, except in the
  deterministic-feedback reversal construction;
- action-quotient equality requires an admissible source-independent lift;
- uniform-integrability failure is necessary for positive-gap collapse, not a
  sufficient characterization;
- the countable theorem is an iid additive theorem for a deliberately
  restricted action class, not a general infinite-product theorem.

Confidence below means confidence in correctness under the displayed
assumptions, not confidence in novelty. “Classical” means that the proof is a
direct application or short synthesis of standard tools; it does not mean that
the exact bit-equivalent formulation has already appeared in print.

## Global definitions and conventions

The finite model is

\[
\mathsf D=(\mathcal T,P_\Theta,\mathcal A,r),\qquad
B_{\mathsf D}(\rho)=
\inf_{K:\,\mathbb E r(\Theta,A)\geq\rho}I(\Theta;A),
\]

where both alphabets are finite, all stochastic kernels are admissible, the
empty infimum is \(+\infty\), and logarithms are natural. The uninformed
baseline is

\[
R_0=\sup_{\nu\perp\Theta}\mathbb E[r(\Theta,A)]
=\sup_a\mathbb E[r(\Theta,a)].
\]

Outside this finite model, every result needs its own measurability,
integrability, compactness, or admissibility contract. Convexity and
monotonicity of the finite frontier are standard consequences of convexity of
mutual information in the channel and nested feasible sets.

## Reward semantics

### R1 — Positive-affine invariance

| Field | Audit |
|---|---|
| Manuscript / technical label | Manuscript Theorem 1 / technical Theorem 1 |
| Exact statement | If \(r'=\alpha r+\beta\), \(\alpha>0\), then \(B_{r'}(\rho)=B_r((\rho-\beta)/\alpha)\). |
| Setting | Any one-shot problem for which the expectations exist; finiteness is not essential. |
| Essential assumptions | The slope is strictly positive; \(\alpha,\beta\) are global constants; the same channel class and information objective are used. |
| Proof location and ingredients | Technical §2, “Theorem 1.” The two reward constraints select exactly the same feasible channels after threshold relabeling. |
| Scope | Static frontier. It also remaps the means of any fixed channel sequence, but does not alone transport an adaptive policy between observation processes. |
| Novelty posture | Classical cardinal expected-utility/rate–distortion invariance; foundational consistency result, not a stand-alone novelty claim. |
| Confidence | **Very high (99%).** |
| Remaining obligation | Cite the closest expected-utility and rate–distortion formulations; keep “conditional mean reward” distinct from transforming a noisy realized reward. |

### R1-S — Sequential affine-transport corollary

| Field | Audit |
|---|---|
| Label | Technical Theorem 1, “Sequential corollary”; summarized in manuscript Theorem 1. |
| Exact statement | If nonreward observations are unchanged and observed realized rewards obey \(R'_t=\alpha R_t+\beta\), policies transport through the inverse affine history map, induce the same marginal action channels, and have identical average bit-equivalent values after threshold relabeling. |
| Setting | Sequential environment with well-defined randomized history-dependent policies. |
| Essential assumptions | Affine reward observations are in bijection; timing and all other observations, dynamics, feasible actions, and policy randomization are unchanged. |
| Proof location and ingredients | Technical §2 immediately after Theorem 1. Bijection of histories followed by R1 at each induced marginal channel. |
| Scope | Sequential, but only a policy-transport statement. It is not an achievability theorem for a different feedback experiment. |
| Novelty posture | Elementary corollary. |
| Confidence | **Very high (97%).** |
| Remaining obligation | In the paper’s formal version, state the history spaces and policy transport recursively; define open-endedness before invoking classification invariance. |

### R2 — Maximal universal pointwise invariance class

| Field | Audit |
|---|---|
| Manuscript / technical label | Manuscript Theorem 2 / technical Theorem 2. |
| Exact statement | For an interval \(J\), continuous strictly increasing \(g:J\to\mathbb R\), and continuous order isomorphism \(h:J\to g(J)\), if \(B^{g\circ r}(h(\rho))=B^r(\rho)\) for every finite problem with rewards in \(J\), then \(h=g\) and \(g(x)=\alpha x+\beta\), \(\alpha>0\). R1 gives the converse. |
| Setting | Universal quantification over finite Bayesian decision problems, including one-state and one-action problems and finite source lotteries. |
| Essential assumptions | Universality over all finite lotteries and a single threshold relabeling independent of the problem are essential. Strict order preservation is essential to the stated interpretation. Continuity is enough for the proof but may be stronger than necessary. |
| Proof location and ingredients | Technical §2, “Theorem 2.” Constant one-action problems force \(h=g\); finite one-action lotteries force Jensen equality; continuity and strict increase yield positive affinity. |
| Scope | Static conditional-mean frontier. It says nothing about transformations tailored to one fixed payoff support. |
| Novelty posture | The mathematical core is classical affine expected-utility uniqueness. The universal frontier formulation is a clean foundations result, but novelty should be framed as application/synthesis. |
| Confidence | **High (96%).** |
| Remaining obligation | Determine the weakest regularity assumptions on \(g,h\); compare the exact universal-conjugacy formulation with utility-representation theorems before claiming novelty. |

### R3 — Invertible monotone reversal of open-endedness

| Field | Audit |
|---|---|
| Manuscript / technical label | Manuscript Theorem 3 / technical Theorem 2.1. |
| Exact statement | In the countable-action deterministic bandit built from iid fair bits, queries, and correct-prefix deployments, \(B^r(\rho)\geq\lambda\rho-\log 2\) for \(0<\lambda<\log2\), and an alternating query/deploy policy has average bit-equivalent \(\Omega(T)\). Under \(g(x)=x^3\), \(B^{r^3}(\rho)=0\) for every finite \(\rho\), although transformed reward feedback is invertibly recoverable. |
| Setting | Source \(\{0,1\}^{\mathbb N}\) with product prior; countable action space; deterministic bandit reward observation; policies have finite per-round mean reward. This is a separately specified standard-Borel extension, not a consequence of finite-problem compactness. |
| Essential assumptions | Unbounded prefix reward and superlinear \(g(n)/n\to\infty\) drive collapse; the query reward reveals each bit; deployment actions explicitly encode a finite prefix; the open-endedness statistic is computed from per-round marginal mean rewards. |
| Proof location and ingredients | Technical §2, “Theorem 2.1.” Donsker–Varadhan with fixed-action exponential moments below 2 gives the lower bound. The alternating policy supplies rewards \(k\). Independent on/off activation of a correct length-\(n\) prefix gives transformed reward \(\rho\) at information \(\rho\log2/n^2\). |
| Scope | Both static and sequential: the collapse and lower bound are frontier statements; the explicit alternating policy supplies the sequential witness. No claim is made for bounded reward or arbitrary bandit protocols. |
| Novelty posture | **Primary candidate contribution.** The ingredients resemble exponential-moment bounds and flash signaling, but the invertible-feedback/open-endedness classification reversal may be new. |
| Confidence | **High (93%).** The construction and constants check out. |
| Remaining obligation | Give the exact imported definition of open-endedness and admissible policy; obtain external review of the Donsker–Varadhan step and conditioning identity; search specifically for ordinal/cardinal counterexamples in rate–distortion, rational inattention, and open-ended learning. |

## Representation

### S1 — Reward-sufficient source reduction

| Field | Audit |
|---|---|
| Manuscript / technical label | Manuscript Theorem 4 / technical Theorem 3. |
| Exact statement | If \(S=s(\Theta)\) and \(r(\theta,a)=\bar r(s(\theta),a)\), then \(B_\Theta(\rho)=B_S(\rho)\) at every threshold. |
| Setting | Finite source and action spaces with all randomized kernels; the displayed proof extends when regular conditional laws exist. |
| Essential assumptions | Exact pointwise reward factorization; deterministic statistic; both lift-through-\(s\) and conditional fiber-averaging kernels must be admissible. Independence of discarded coordinates is **not** required for the static equality. |
| Proof location and ingredients | Technical §3, “Theorem 3.” Lift an \(S\)-channel to obtain \(I(\Theta;A)=I(S;A)\); average a \(\Theta\)-channel conditional on \(S\) to preserve \((S,A)\) and use data processing. |
| Scope | Static optimized frontier. Correlated discarded variables may remain useful observations during acquisition, so this does not equate learning trajectories. |
| Novelty posture | Classical statistical sufficiency/data processing, specialized to reward profiles. Likely a semantic-cleanup theorem rather than a novel information-theory result. |
| Confidence | **Very high (99%)** in the finite setting. |
| Remaining obligation | State a standard-Borel version with a regular conditional distribution and an explicitly closed admissible kernel class; cite Bayesian sufficiency and rate–distortion source reduction. |

### A1 — Behavioral action quotient

| Field | Audit |
|---|---|
| Manuscript / technical label | Manuscript Theorem 5 / technical Theorem 4. |
| Exact statement | If \(r(\theta,a)=\bar r(\theta,q(a))\), pushing forward gives \(B_{\bar{\mathcal A}}(\rho)\leq B_{\mathcal A}(\rho)\). Equality holds when every quotient action has a source-independent admissible stochastic lift supported on its \(q\)-fiber; in the unrestricted finite setting, a section supplies one. |
| Setting | Finite unrestricted kernels as the base theorem; measurable/resource/feasibility-preserving lift required in general spaces or constrained models. |
| Essential assumptions | Exact reward preservation for the pushforward inequality. Surjectivity onto the quotient’s admissible actions and a source-independent admissible lift for the reverse inequality. |
| Proof location and ingredients | Technical §3, “Theorem 4.” Data processing under \(A\mapsto q(A)\); for the reverse direction use \(\Theta\to\bar A\to A\) and \(\bar A=q(A)\), obtaining equality of mutual informations. |
| Scope | Static optimized frontier. It does not say that every raw representation or raw policy has the same information; optimization can discard serialization bits. |
| Novelty posture | Classical data processing and quotienting. The useful contribution is identifying the exact lift condition for the bit-equivalent. |
| Confidence | **Very high (98%)** under the stated lift contract. |
| Remaining obligation | Formalize “admissible” for any intended dynamics/resources; prove or assume measurable selection outside finite spaces; keep the one-sided inequality separate from equality. |

### A2 — Experiment-restricted Blackwell monotonicity

| Field | Audit |
|---|---|
| Label | Technical Proposition 4.1; prose following manuscript Theorem 5. |
| Exact statement | If experiment \(E_1\) Blackwell-dominates \(E_2\), then the frontier restricted to decision kernels based on the experiment satisfies \(B_{E_1}(\rho)\leq B_{E_2}(\rho)\); Blackwell-equivalent experiments have equal restricted frontiers. |
| Setting | Fixed payoff state and reward, randomized decision rules, and information objective \(I(W;A)\); the unrestricted frontier is the identity-experiment case. |
| Essential assumptions | The more informative experiment can simulate the less informative one, and the simulated decision rule remains admissible. |
| Proof location and ingredients | Technical §3, Proposition 4.1. Simulation reproduces the joint law of \((W,A)\). |
| Scope | Static and experiment-restricted; not a theorem that arbitrary source encodings are Blackwell-equivalent. |
| Novelty posture | Directly classical Blackwell comparison. Context only. |
| Confidence | **Very high (98%).** |
| Remaining obligation | Give a precise definition of \(B_E\) if retained in the final paper and cite Blackwell directly. |

## Nondegeneracy and collapse

### N0 — Attained zero information

| Field | Audit |
|---|---|
| Label | Technical Lemma 5.0; used throughout manuscript §5. |
| Exact statement | A feasible channel attains information zero at \(\rho\) iff a source-independent action distribution attains reward at least \(\rho\). In a finite action space this is equivalent to a constant action doing so. For general spaces, every \(\rho<R_0\) has such a witness; \(\rho=R_0\) does iff the supremum is attained. |
| Setting | Any setting where mutual information and expected reward are well defined; the constant-action equivalence uses finite actions. |
| Essential assumptions | “Attains” cannot be replaced by “has infimum zero.” Boundary attainment requires compactness/upper semicontinuity or a direct maximizer assumption. |
| Proof location and ingredients | Technical §4, Lemma 5.0. \(I=0\) iff the joint law factors; a linear objective over finite action mixtures is maximized by a component. |
| Scope | Static. |
| Novelty posture | Elementary. Its value is preventing the infimum/minimum ambiguity. |
| Confidence | **Very high (99%).** |
| Remaining obligation | Keep four cases distinct: attained zero, boundary nonattainment at \(R_0\), positive-gap collapse above \(R_0\), and infeasibility. Only the positive-gap case forces failure of the stated two-law uniform-integrability condition. |

### N1 — Bounded positive-gap certificate

| Field | Audit |
|---|---|
| Manuscript / technical label | Manuscript Theorem 6 / technical Theorem 5. |
| Exact statement | If \(r\in[m,M]\), \(L=M-m>0\), then for \(\rho>R_0\), \(B(\rho)\geq2((\rho-R_0)/L)^2\). |
| Setting | Any one-shot problem with globally bounded reward and well-defined channels; finiteness is not used in the inequality itself. |
| Essential assumptions | A uniform oscillation bound and a strictly positive gap above \(R_0\). The bound is in nats and uses the stated total-variation convention. |
| Proof location and ingredients | Technical §4, “Theorem 5.” Compare the induced joint law \(P\) to \(Q=P_\Theta P_A\); bounded oscillation controls expectation difference by \(L\|P-Q\|_{TV}\); Pinsker gives \(\|P-Q\|_{TV}\leq\sqrt{I/2}\). |
| Scope | Static lower bound. It neither supplies a learner nor resolves the boundary \(\rho=R_0\). |
| Novelty posture | Classical Pinsker certificate; likely application/synthesis. |
| Confidence | **Very high (99%).** |
| Remaining obligation | Verify constants against the final log and TV conventions; state separately that strict positivity at the boundary needs attainment/compactness. |

### N2 — Tail escape is necessary for positive-gap collapse

| Field | Audit |
|---|---|
| Manuscript / technical label | Manuscript Theorem 7 / technical Theorem 6. |
| Exact statement | If \(I(P_n)=D(P_n\|Q_n)\to0\), \(Q_n=P_\Theta P_{A,n}\), and \(r\) is uniformly integrable under both \(\{P_n\}\) and \(\{Q_n\}\), then \(\mathbb E_{P_n}r-\mathbb E_{Q_n}r\to0\). Therefore a sequence with reward at least fixed \(\rho>R_0\) and information tending to zero must violate that two-law UI condition. |
| Setting | Varying joint laws and their independence references; unbounded signed reward allowed subject to the explicit two-family UI condition. |
| Essential assumptions | Vanishing KL/mutual information and tail control under **both** law families. The conclusion is necessary-only: UI failure does not itself imply collapse. |
| Proof location and ingredients | Technical §4, “Theorem 6.” Pinsker gives TV convergence; truncate at \(\pm K\), control the bounded part by \(2K\,TV\), then send \(n\to\infty\) and \(K\to\infty\). |
| Scope | Static sequence criterion. |
| Novelty posture | Classical UI-plus-total-variation lemma applied to the frontier. The precise collapse diagnosis may be useful, but should not be sold as a complete taxonomy. |
| Confidence | **Very high (97%).** |
| Remaining obligation | Check for known “convergence in total variation plus UI” formulations; decide whether useful one-sided or Orlicz-envelope variants can be proved. |

### N3 — Rare-burst collapse mechanism

| Field | Audit |
|---|---|
| Label | Technical Lemma 6.1; examples summarized in manuscript §5. |
| Exact statement | If a baseline has reward \(b\), and channels have reward \(R_n>b\), information \(J_n\), with \(R_n\to\infty\) and \(J_n/(R_n-b)\to0\), independent activation at probability \((\rho-b)/(R_n-b)\) proves \(B(\rho)=0\) for every finite \(\rho>b\). |
| Setting | A common action model in which baseline/burst randomization is admissible and expected rewards exist. |
| Essential assumptions | Independent activation; eventually \(R_n\geq\rho\); a baseline action; sublinear information per excess reward. Disjoint supports are needed only for equality in the information calculation, not for the upper bound. |
| Proof location and ingredients | Technical §4, Lemma 6.1. Adjoin the activation coin, use its independence and data processing to get \(I\leq dJ_n\). Strict-margin and small-correlation examples instantiate the mechanism or a related leverage mechanism. |
| Scope | Static counterexample generator. |
| Novelty posture | Analogous to flash signaling/capacity-per-unit-cost constructions; its use against the bit-equivalent may be new. |
| Confidence | **Very high (98%).** |
| Remaining obligation | Compare terminology and exact ratios with flash-signaling literature; do not imply rare activation is the only collapse mechanism. |

### N4 — Compact bounded-reward attainment

| Field | Audit |
|---|---|
| Label | Technical Theorem 6.2; prose in manuscript §5. |
| Exact statement | For a Polish source with fixed prior, compact metric action space, all Borel kernels, and bounded continuous reward, every feasible frontier infimum is attained, \(R_0\) is attained, and \(B(\rho)=0\) iff \(\rho\leq R_0\). |
| Setting | Standard-Borel/Polish one-shot decision problem represented by joint laws with fixed source marginal. |
| Essential assumptions | Tight weakly closed joint-law class, closed reward constraint, lower semicontinuity of mutual information, and attainment of \(R_0\). Compactness plus bounded continuity are convenient sufficient conditions; constrained kernel classes must also be weakly closed. |
| Proof location and ingredients | Technical §4, Theorem 6.2. Prokhorov compactness, continuity of product marginals, lower semicontinuity of relative entropy, and continuity of expected reward. |
| Scope | Static existence/noncollapse theorem. |
| Novelty posture | Standard variational/topological argument. |
| Confidence | **High (94%).** |
| Remaining obligation | Have a measure-theory reviewer check compactness of the fixed-marginal law class and continuity of \(P_A\mapsto P_\Theta\otimes P_A\); state exact topology for restricted kernels. |

## Composition

### C1 — Finite independent-product infimal convolution

| Field | Audit |
|---|---|
| Manuscript / technical label | Manuscript Theorem 8 / technical Theorem 7. |
| Exact statement | For finitely many independent sources, product actions, and additive reward, \(B_{\otimes_i\mathsf D_i}(\rho)=\inf_{\sum_i\rho_i\geq\rho}\sum_iB_i(\rho_i)\). |
| Setting | Finite source/action problems with all product-action kernels; extended-real infima. |
| Essential assumptions | Source independence for the information lower bound; additive reward for scalar allocation; product action feasibility for achievability. The global channel may correlate actions across coordinates. |
| Proof location and ingredients | Technical §5, “Theorem 7.” Independence and conditional-entropy subadditivity give \(I(\Theta_{1:n};A_{1:n})\geq\sum_iI(\Theta_i;A_i)\); component frontier bounds give the converse; product near-minimizers give achievability. |
| Scope | Static. It does not construct a sequential learner. |
| Novelty posture | Classical separable rate–distortion structure. Use as a consistency theorem. |
| Confidence | **Very high (98%)** in the finite setting. |
| Remaining obligation | Specify extended-real allocation conventions and cite classical separable rate–distortion/tensorization results; use relative entropy rather than finite entropy before extending source alphabets. |

### C2 — Identical finite tensorization

| Field | Audit |
|---|---|
| Manuscript / technical label | Part of manuscript Theorem 8 / technical Corollary 7.1. |
| Exact statement | For \(n\) identical independent components under C1, \(B_n(\rho)=nB_1(\rho/n)\). |
| Setting | C1 with identical components. |
| Essential assumptions | C1 plus convexity and monotonicity of the component frontier; equal component domains. |
| Proof location and ingredients | Technical §5, Corollary 7.1. Jensen makes equal reward allocation optimal. |
| Scope | Static finite product. |
| Novelty posture | Classical. |
| Confidence | **Very high (98%).** |
| Remaining obligation | State how infeasible thresholds and \(+\infty\) values are handled in the Jensen argument. |

### C3 — Countable iid local-price law

| Field | Audit |
|---|---|
| Manuscript / technical label | Manuscript Theorem 9 / technical Theorem 7.2. |
| Exact statement | For a finite component problem with a pointwise zero-reward null action and positive finite maximal reward, take iid countably many sources. Restrict global actions to finite support almost surely and require \(\mathbb EN(A)<\infty\). With additive reward and \(\kappa=\lim_{x\downarrow0}B_1(x)/x\), \(B_\infty(\rho)=\kappa\rho\) for every finite \(\rho\geq0\). |
| Setting | Countable iid product source; countable finite-support action space; bounded finite-component reward; all admissible kernels satisfying finite expected support. |
| Essential assumptions | Iid source independence; identical component frontier; additive reward; pointwise null action; an absolute-integrability condition (here implied by finite expected support); unrestricted use of any finite number of coordinates for achievability. |
| Proof location and ingredients | Technical §5, Theorem 7.2. Convexity gives \(B_1(x)\geq\kappa x\). Finite-coordinate projection and C1 give the lower bound; finite expected support gives absolute reward summability and Fubini. Equal allocation over the first \(n\) coordinates gives the upper bound. |
| Scope | Static countable-product frontier. It does not establish a schedule-free agent, non-iid composition, or arbitrary infinite-support admissibility. |
| Novelty posture | **Secondary candidate contribution.** Closely related results may exist in separable/countable rate–distortion and capacity-per-unit-cost theory; novelty is unresolved. |
| Confidence | **High (91%).** The written proof is coherent; its exact literature status and weakest integrability conditions need review. |
| Remaining obligation | External information-theory review; exhaustive search for countable separable rate–distortion formulas; make extended mutual information and absolute convergence conventions explicit; determine whether \(\mathbb EN<\infty\) can be replaced exactly by \(\mathbb E\sum_i|r_i|<\infty\) plus a suitable information limit. |

## Submission gate

The mathematics is ready for full-paper drafting, but not yet for an
unqualified novelty claim. Before submission:

1. mechanically verify all finite claims on enumerated small problems;
2. retain the counterexamples in `assumption-stress-tests.md` beside the proof
   source while editing;
3. have an information theorist independently review R3 and C3;
4. have a measure-theory reviewer check N4 and any standard-Borel extension;
5. complete targeted prior-art searches before labeling R3 or C3 “new”; and
6. keep static frontier geometry separate from sequential attainability in the
   abstract, theorem statements, and conclusion.
