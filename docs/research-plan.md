# Open-Ended or Just Novel?
## Implementation-Ready Research and Experimental Specification

**Working title:** *Open-Ended or Just Novel? An Experimental Test of Reward-Relevant Information*

**Document status:** Version 2 — implementation started

**Primary source:** Wanqiao Xu, Yifan Zhu, and Benjamin Van Roy, [*An Information-Theoretic Definition for Open-Ended Learning*](https://arxiv.org/abs/2606.08369), 2026.

**Purpose:** This document preserves the full ambition of the original research plan while resolving the mathematical, measurement, protocol, and engineering ambiguities that would otherwise make the results difficult to interpret.

---

## 1. Executive decision

The project will remain a broad experimental validation of the bit-equivalent concept. It will include:

1. an analytically solvable stationary symbolic benchmark;
2. exact finite rate-distortion solvers;
3. direct and variational large-instance estimators;
4. cumulative, redundant, irrelevant, public-reward, opportunity, and ephemeral controls;
5. reward-, novelty-, and information-directed agents;
6. fixed, scheduled, unrestricted, and adaptive learning targets;
7. bounded and nonlinear reward stress tests;
8. behavioral action-representation tests;
9. scaling-law inference;
10. and a procedural neural extension.

The scope is not reduced. It is divided into gated workstreams so that later results are interpretable even if an ambitious estimator or neural extension fails.

The core paper claim is:

> The bit-equivalent is an empirically meaningful measure of cumulative capability growth when it is computed over behavioral actions, calibrated against exact decision frontiers, and separated from the process by which an agent acquires information.

The project must be willing to reject or sharply qualify that claim if the estimator leaks distractor information, depends on action serialization, or cannot reproduce known frontiers.

### 1.1 Research questions and registered hypotheses

**RQ1 — Measurability.** Can the reward-information frontier be computed or bounded tightly enough to support empirical conclusions?

**H1:** Exact solvers agree with analytic controls; approximate estimators produce bounds with frozen coverage targets and widths narrow enough to distinguish the registered controls.

**RQ2 — Irrelevance.** Does the metric ignore information that cannot improve reward?

**H2:** ALEA raises reward-irrelevant novelty, while TRIVIA raises acquired persistent irrelevant information; neither changes the exact reward-information frontier, and calibrated approximate estimates remain invariant within frozen tolerances.

**RQ3 — Independence and redundancy.** Does the metric respond to effective reward-relevant rank rather than task count?

**H3:** IND produces an unbounded linear frontier. Unrestricted additive RED exhibits the predicted zero-information rare-burst collapse, resource-controlled RED remains bounded by its reward-sufficient latent core, and MIX agrees with its exact or certified reward-information frontier.

**RQ4 — Accumulation and turnover.** Is continual adaptation sufficient for open-endedness?

**H4:** EPH supports continual acquisition while its current-decision bit-equivalent remains bounded by active-state complexity.

**RQ5 — Exploration objective.** Do novelty and total information target useful knowledge?

**H5:** Novelty- and total-information-directed agents acquire more irrelevant information and exhibit lower useful-information efficiency than reward- or relevance-directed agents under matched budgets.

**RQ6 — Target expansion.** Is an expanding learning target necessary for sustained useful-information growth?

**H6:** For the registered allocation rules, fixed targets saturate, suitably expanding targets sustain the analytic growth regime, and overly aggressive expansion can increase frontier regret under bounded acquisition.

**RQ7 — Adaptive curricula.** Can target expansion be selected without a hand-tuned growth schedule?

**H7:** Frontier-following curricula match a tuned schedule across multiple noise, alphabet, and reward-margin settings. Target-discovery performance is evaluated separately from oracle-assisted scheduling.

**RQ8 — Reward semantics.** Which reward transformations preserve classification?

**H8:** Positive affine transforms obey exact frontier remapping. Nonlinear transforms can alter the scaling class and are sensitivity tests rather than invariance hypotheses.

**RQ9 — Action semantics.** Is the result about behavior or serialization?

**H9:** Symbolic representations that compile to the same finite behavior agree exactly. Neural or sampled behavioral approximations agree only within calibrated tolerances.

**RQ10 — Acquisition difficulty.** Does noisier feedback change the metric or only the learning curve?

**H10:** Holding prior, action, and reward fixed, feedback noise changes acquisition speed but leaves the decision frontier unchanged.

**RQ11 — Generalization.** Does the estimator work beyond separable IID rules?

**H11:** Certified or calibrated bounds cover held-out mixed-structure and dependency-graph frontiers and are narrow enough to recover the registered ordering.

**RQ12 — Procedural transfer.** Do the central distinctions survive function approximation and composition?

**H12:** The ordering of IND, RED, and TRIVIA survives in the procedural wrapper as transfer evidence, subject to reported behavioral-equivalence and estimator uncertainty.

### 1.2 Alignment with the source definition

| Source-paper element | This project |
|---|---|
| Environment uncertainty is a prior over \(\Theta\) | The full Rulebook latent is sampled once from a declared prior |
| \(B_\rho\) minimizes \(I(\Theta;A)\) over actions attaining pooled expected mean reward | The primary frontier is solved over canonical behavioral deployment channels |
| Open-endedness means \(\overline B_T=\Omega(T)\) for some agent | Scaling tests use the same average-bit-equivalent target and clearly label finite-horizon evidence |
| The insatiable action is an infinite vector with finite support | A Rulebook deployment is countably indexed with finite support and finite expected support |
| The source paper’s primary interaction is a stationary bandit | P0 preserves strict aggregate-bandit alignment |
| The source paper identifies bandit-only scope as a limitation | P1, P2, EPH, and the procedural world are labeled extensions |
| The source algorithm is given a sequence of learning targets | Oracle scheduling, adaptive scheduling, and target discovery are evaluated separately |
| The source gives a bounded logistic open-ended example | Logistic and other bounded rewards are treated as formal reward-semantic experiments |

---

## 2. Corrections that are now locked

These are design requirements, not optional ablations.

### 2.1 The uninformed prediction margin is strictly negative

For a \(q\)-ary rule, correct deployment earns \(u>0\), incorrect deployment earns \(-c<0\), and abstention earns zero. Require

\[
c>\frac{u}{q-1}.
\]

Equivalently, an uninformed prediction has a fixed negative expected score:

\[
\frac{u}{q}-c\left(1-\frac{1}{q}\right)<0.
\]

Equality is forbidden. At equality, infinitesimally better-than-chance predictions yield reward linear in their advantage but mutual information quadratic in that advantage. Spreading a fixed reward across more coordinates then drives the required information to zero. The intended positive control would collapse.

The baseline parameters are:

\[
q=4,\qquad u=1,\qquad c=1.
\]

An uninformed prediction then earns \(-1/2\) in expectation, and deployment is worthwhile only when posterior correctness exceeds \(1/2\).

For alphabet sweeps, parameterize the cost through the profitability threshold

\[
\tau=\frac{c}{u+c},
\qquad
c=\frac{u\tau}{1-\tau},
\]

and require \(\tau>1/q\). Do not reuse \(u=c=1\) for the binary case, where it would recreate the forbidden equality margin.

### 2.2 The environment is stationary

All countably many primitive rules exist from time zero. Rules are generated lazily in software, but not created over time in the mathematical environment.

Phrases such as “new rules arrive” will mean one of:

- the agent queries a previously untouched rule;
- the agent adds a rule to its learning target;
- the agent expands its finite deployment;
- or, only in the explicitly labeled ephemeral extension, a dynamic latent slot refreshes.

The cumulative benchmark itself never changes its latent law, reward function, or action space with time.

### 2.3 Acquisition and evaluation are separate

Training interactions acquire information through a bounded-bandwidth interface. Checkpoint evaluation freezes a deployment and returns no observations to the agent.

This prevents an expanding rulebook from receiving expanding per-rule feedback and prevents evaluation from teaching the agent.

### 2.4 The primary information quantity is \(I(\Theta;A)\)

The theoretical definition minimizes mutual information between the environment and the behavioral action:

\[
B_\rho=\inf_{P(A\mid\Theta):\,\mathbb E[r_\Theta(A)]\ge\rho}I(\Theta;A).
\]

A latent bottleneck \(Z\) is a computational device, not the definition. Penalizing \(I(\Theta;Z)\) generally supplies only an achievable upper bound because

\[
I(\Theta;A)\le I(\Theta;Z)
\]

for \(\Theta\to Z\to A\).

Whenever a latent estimator is used, the induced action channel

\[
P(A\mid\Theta)=\sum_z q(z\mid\Theta)\pi(A\mid z)
\]

must be analyzed, and the distinction between \(I(\Theta;A)\) and \(I(\Theta;Z)\) must be reported.

### 2.5 Actions are identified by behavior

The primary action is a canonical behavioral rulebook, not a byte string, parameter vector, syntax tree, or neural checkpoint.

Equivalent tables, programs, and policies are mapped to the same induced behavior whenever the domain permits exact canonicalization. Approximate behavioral equivalence in the neural extension is explicitly labeled as approximate.

General program and neural equivalence is not decidable. Symbolic program actions are accepted only when they compile, within declared resource limits, to a terminating finite canonical rulebook on the registered projection. Infinite-support, nonterminating, or unresolved programs are infeasible actions rather than new serializations.

### 2.6 Infinite entropy is never evaluated as \(\infty-\infty\)

For infinite \(\Theta\), information gain is defined as

\[
I(\Theta;H_t)
=
\mathbb E_{H_t}
D_{\mathrm{KL}}\!\left(
P(\Theta\mid H_t)\,\|\,P(\Theta)
\right)
=
\sup_N I(\Theta_{1:N};H_t).
\]

In the finite-query factorized case this becomes a finite coordinatewise sum. The implementation must never calculate

\[
H(\Theta)-H(\Theta\mid H_t)
\]

when both terms are infinite.

### 2.7 Information units are nats

All logarithms are natural logarithms, matching the source paper. Figures may include a secondary bits scale using division by \(\log 2\), but stored metrics and solver tolerances are in nats.

---

## 3. Formal research object

### 3.1 Static environment

The master environment has a latent variable \(\Theta\) sampled once from a declared prior and public surface-rule maps

\[
Y_i=f_i(\Theta)\in\{1,\ldots,q\}.
\]

Each condition specifies the minimal latent representation and the maps \(f_i\). Surface labels are marginally uniform unless a condition explicitly tests a different prior.

In IND only,

\[
\Theta=(\Theta_1,\Theta_2,\ldots),
\qquad
Y_i=\Theta_i,
\qquad
\Theta_i\overset{\mathrm{iid}}{\sim}\mathrm{Uniform}\{1,\ldots,q\}.
\]

The latent is sampled once. A software implementation generates independent primitive components using a counter-based pseudorandom function keyed by the environment seed, so values are invariant to query order, parallel execution, or checkpointing.

### 3.2 Canonical deployment action

A deployment is

\[
A=(A_1,A_2,\ldots),
\qquad
A_i\in\{0,1,\ldots,q\},
\]

where \(A_i=0\) means abstain and

\[
\|A\|_0=|\{i:A_i\ne0\}|<\infty.
\]

Randomized deployment kernels must satisfy

\[
\mathbb E[\|A\|_0]<\infty
\]

so expected reward is well defined.

Duplicate indices, serialization order, comments, unused program state, and parameter symmetries are removed by canonicalization.

### 3.3 Additive mean reward

\[
r_\theta(A)
=
\sum_{i:A_i\ne0}
\left[
u\,\mathbf 1\{A_i=f_i(\theta)\}
-
c\,\mathbf 1\{A_i\ne f_i(\theta)\}
\right].
\]

Abstention contributes zero. The baseline uses \(q=4,u=c=1\).

### 3.4 Bit-equivalent

\[
B_\rho
=
\inf_{P(A\mid\Theta):\,
\mathbb E[r_\Theta(A)]\ge\rho}
I(\Theta;A).
\]

For the infinite product environment, define

\[
I(\Theta;A)
:=
\sup_N I(\Theta_{1:N};A),
\]

which is the mutual information generated by the increasing finite-coordinate sigma-algebras.

The infimum ranges over stochastic behavioral channels. It is not restricted to channels an implemented learner can acquire within a given sample budget.

This distinction is essential:

- \(B_\rho\) measures the least decision information compatible with performance;
- the learning curve measures whether an agent can acquire and use that information;
- sample complexity measures how costly acquisition is;
- compute measures how costly inference and optimization are.

### 3.5 Average bit-equivalent

For pooled expected reward

\[
\rho_t=\mathbb E_{\Theta,\pi}[r_\Theta(A_t)],
\]

define

\[
\overline B_T(\pi)
=
\frac1T\sum_{t=0}^{T-1}B_{\rho_t}.
\]

The order of operations is fixed:

\[
B_{\mathbb E[R_t]},
\]

not

\[
\mathbb E[B_{R_t\mid\text{seed}}].
\]

The two can differ because the frontier is nonlinear.

---

## 4. Exact one-coordinate frontier

The one-coordinate derivation is the mathematical foundation and the first implementation gate.

This section applies to the IND source \(Y_i=\Theta_i\) with a uniform \(q\)-ary prior.

### 4.1 Symmetric channel

By averaging any channel over simultaneous permutations of source and prediction labels, expected reward is preserved and mutual information cannot increase. An optimal channel can therefore be represented by:

- \(d\in[0,1]\): probability of deployment;
- \(p\in[1/q,1]\): correctness conditional on deployment.

For \(\Theta\sim\mathrm{Uniform}[q]\):

\[
P(A=0\mid\Theta)=1-d,
\]

\[
P(A=\Theta\mid\Theta)=dp,
\]

\[
P(A=j\ne\Theta\mid\Theta)
=
d\frac{1-p}{q-1}.
\]

Define conditional value

\[
g(p)=(u+c)p-c
\]

and conditional information

\[
J_q(p)
=
p\log(qp)
+
(1-p)\log\left(\frac{q(1-p)}{q-1}\right).
\]

Then

\[
R(d,p)=d\,g(p),
\qquad
I(d,p)=d\,J_q(p).
\]

Profitable deployment requires

\[
p>\tau:=\frac{c}{u+c}>\frac1q.
\]

### 4.2 Exact formula

For \(0<r\le u\),

\[
B_1(r)
=
\min_{p\ge(r+c)/(u+c)}
r\,\frac{J_q(p)}{g(p)},
\qquad
B_1(0)=0.
\]

Let \(p^\star\) minimize \(J_q(p)/g(p)\) over \(p\in(\tau,1]\). The interior solution satisfies

\[
g(p^\star)J_q'(p^\star)
=(u+c)J_q(p^\star),
\]

where

\[
J_q'(p)=
\log\left(\frac{(q-1)p}{1-p}\right).
\]

Define

\[
\kappa=\frac{J_q(p^\star)}{g(p^\star)},
\qquad
r^\star=g(p^\star).
\]

Then

\[
B_1(r)=
\begin{cases}
\kappa r, & 0\le r\le r^\star,\\[4pt]
J_q\!\left(\dfrac{r+c}{u+c}\right),
&r^\star\le r\le u,\\[8pt]
+\infty,&r>u.
\end{cases}
\]

### 4.3 Closed-form baseline

For \(q=4,u=c=1\):

\[
\tau=\frac12,
\qquad
p^\star=\frac34,
\qquad
r^\star=\frac12,
\qquad
\kappa=\log 3.
\]

Thus

\[
B_1(r)=
\begin{cases}
r\log 3,&0\le r\le 1/2,\\[4pt]
J_4\!\left(\dfrac{r+1}{2}\right),&1/2\le r\le1.
\end{cases}
\]

This closed form is the principal numerical regression oracle.

### 4.4 Numerical solution

For general \(q,u,c\), solve for \(p^\star\) by bisection on

\[
F(p)=g(p)J_q'(p)-(u+c)J_q(p).
\]

Since

\[
F'(p)=g(p)J_q''(p)>0
\]

on the profitable interval, the root is unique when interior.

Required unit tests:

- \(B_1(0)=0\);
- \(B_1(u)=\log q\);
- continuity at \(r^\star\);
- matching left and right derivatives at \(r^\star\);
- monotonicity;
- convexity;
- and agreement with the finite direct-channel solver.

---

## 5. Tensorization and the infinite frontier

### 5.1 Finite \(N\)

For \(N\) independent coordinates with additive reward:

\[
B_N(\rho)
=
\inf_{\sum_i r_i\ge\rho}
\sum_{i=1}^N B_1(r_i).
\]

Convexity and exchangeability give

\[
B_N(\rho)
=
N B_1\!\left(\frac{\rho}{N}\right),
\qquad
0\le\rho\le Nu.
\]

The converse follows from source independence, the entropy chain rule, and data processing:

\[
I(\Theta_{1:N};A_{1:N})
\ge
\sum_{i=1}^N I(\Theta_i;A_i).
\]

Achievability uses independent product channels.

### 5.2 Infinite stationary Rulebook

For any finite \(\rho\), choose enough coordinates that the reward assigned to each lies in the linear portion of \(B_1\). Then

\[
B_\infty(\rho)=\kappa\rho.
\]

For the baseline:

\[
B_\infty(\rho)=\rho\log3.
\]

Therefore, if an agent attains

\[
\rho_t=\Theta(t),
\]

then

\[
B_{\rho_t}=\Theta(t)
\]

and

\[
\overline B_T=\Theta(T).
\]

### 5.3 Interpretation caveat

The frontier matches expected reward, not “number of perfectly memorized rules.” A learner that perfectly memorizes \(m\) rules can contain much more environment information than \(B_{mu}\), because a lower-information stochastic policy may attain the same expected reward by spreading moderate confidence across more rules.

This is faithful to the definition, not an estimator error. The paper will report:

- perfectly mastered rules;
- posterior confidence distribution;
- deployed coverage;
- reward variance and lower-tail reward;
- actual relevant information in history;
- and bit-equivalent.

A separate risk-constrained or support-constrained frontier may be included as an explicitly different secondary construct. It must never be substituted silently for \(B_\rho\).

---

## 6. The zero-margin collapse

The paper should state and test this pathology because it is an important empirical design lesson.

### 6.1 Positive uninformed reward

If

\[
c<\frac{u}{q-1},
\]

uninformed guesses have positive value. By deploying sufficiently many independent guesses, arbitrary finite expected reward is attainable with zero information:

\[
B_\infty(\rho)=0.
\]

### 6.2 Zero uninformed reward

If

\[
c=\frac{u}{q-1},
\]

write \(p=1/q+\delta\). As \(\delta\downarrow0\),

\[
g(p)=\Theta(\delta),
\qquad
J_q(p)=\Theta(\delta^2).
\]

Spreading a fixed target reward across \(N\) coordinates gives total information \(O(1/N)\), hence again

\[
B_\infty(\rho)=0.
\]

### 6.3 Experimental use

The equality case will be retained as an adversarial metric test. The expected result is frontier collapse, not open-endedness. It is not a valid primary reward setting.

---

## 7. Interaction protocols

The project uses three protocols. Results are labeled by protocol.

### 7.1 Protocol P0: strict aggregate-bandit alignment

At round \(t\), the agent selects a finite deployment \(A_t\) and receives

\[
R_{t+1}=r_\Theta(A_t)+W_{t+1}
\]

with fixed scalar noise.

This most closely matches the source paper. It is computationally difficult because aggregate feedback couples the posterior over deployed coordinates.

Fixed-noise scalar feedback is not automatically a bounded-information channel when deployment support and reward amplitude are unbounded. P0 experiments must either:

- cap training-time deployment support or reward power;
- or measure and report
  \[
  I(\Theta;R_{t+1}\mid H_t,A_t)
  \]
  rather than claim bandwidth matching with P1.

Uncapped P0 is labeled variable- or unbounded-bandwidth.

Uses:

- formal source alignment;
- aggregate-feedback robustness;
- particle and variational posterior experiments;
- reproduction of expanding-target phenomena under a scalar channel.

### 7.2 Protocol P1: bounded-query construct-validation protocol

At round \(t\), the agent selects an acquisition action

\[
Q_t\subset\mathbb N,
\qquad
|Q_t|\le b.
\]

For each \(i\in Q_t\), it receives a \(q\)-ary symmetric observation:

\[
P(O=Y_i)=1-\epsilon,
\]

\[
P(O=j\ne Y_i)=\frac{\epsilon}{q-1},
\qquad
\epsilon<\frac{q-1}{q}.
\]

P1 queries noisy surface labels, not privileged latent-core coordinates. Direct core queries are a separately labeled oracle condition.

Deployments are frozen and evaluated at checkpoints without feedback. This is a stationary partially observed decision process rather than the source paper’s strict reward-only bandit.

Uses:

- exact posterior updates;
- clean separation of acquisition and decision information;
- agent and distractor comparisons;
- controlled information-budget experiments.

The per-query channel capacity is

\[
C_\epsilon
=
\log q-h_2(\epsilon)-\epsilon\log(q-1).
\]

Thus

\[
I(\Theta;H_t)\le btC_\epsilon,
\]

which supplies an implementation sanity bound. In P1, linear per-round bit-equivalent is the fastest feasible asymptotic rate under fixed \(b\), provided query observations are the agent’s only \(\Theta\)-dependent input. Public labels, pretrained environment-dependent state, or other side information must be included in \(H_t\) or invalidate this ceiling.

### 7.3 Protocol P2: bounded semi-bandit feedback

The agent deploys a rulebook but receives correctness feedback for at most \(b\) prespecified or sampled entries.

It is forbidden to return per-rule feedback for every entry of an expanding deployment. Feedback bandwidth must not scale with rulebook support.

The primary P2 channel is locked as follows:

- before observing the outcome, the agent submits a deployment \(A_t\) and feedback subset \(F_t\subseteq\operatorname{supp}(A_t)\);
- \(|F_t|\le b\);
- \(F_t\) is part of the recorded training action and history;
- feedback noise is conditionally independent across selected entries;
- the aggregate reward and selected correctness observations are returned together.

A separate ablation samples \(F_t\) uniformly from deployment support using environment randomness revealed only with the observation. The two channels are never pooled.

### 7.4 Side-effect-free checkpoints

Checkpoint evaluation:

- freezes the agent before evaluation;
- draws from an independent evaluation seed stream;
- returns no feedback to the learner;
- causes no posterior update;
- causes no environment-state mutation;
- consumes no training RNG state;
- and cannot change future queried coordinates.

For the symbolic environment, mean reward should be computed analytically from the agent’s deployment distribution whenever possible.

Otherwise, evaluate frozen deployment samples against the true latent environment, average over action randomness, and only then pool across environment seeds. The agent’s posterior is never used as ground-truth reward. Record deployment-kernel seed and action-sample count.

---

## 8. Analytic end-to-end sanity agent

Before implementing sophisticated learners, implement a policy with a known growth rate.

If a single noisy observation has accuracy

\[
1-\epsilon>\tau,
\]

the agent:

1. queries \(b\) fresh coordinates per round;
2. stores the observed label;
3. deploys all stored predictions.

Each queried coordinate contributes expected reward

\[
s=(u+c)(1-\epsilon)-c>0.
\]

Therefore

\[
\rho_t=bts,
\qquad
B_{\rho_t}=\kappa bts,
\]

and

\[
\overline B_T
=
\frac{\kappa bs}{2}T+O(1).
\]

If one observation is insufficient, use either:

- a fixed number \(L\) of repeated queries followed by MAP deployment only on histories whose posterior confidence exceeds \(\tau\), choosing \(L\) so the thresholded policy has strictly positive expected reward;
- or a stopping rule with finite expected query cost \(\mathbb E[L]\).

The resulting rate remains \(\Theta(T)\), with the coefficient derived from \(L\) or \(\mathbb E[L]\), respectively.

This test connects:

- latent generation;
- noisy feedback;
- posterior updates;
- deployment;
- reward evaluation;
- frontier inversion;
- and average bit-equivalent.

It must pass before baseline-agent sweeps begin.

---

## 9. Master environment family

Primary controls share:

- the same \(q\);
- the same query budget;
- the same feedback channel;
- the same deployment representation;
- the same abstention and score semantics;
- the same task-index interface;
- and matched attainable score ranges wherever mathematically possible.

### 9.1 Independent cumulative rules: IND

\[
Y_i=Z_i,
\qquad
Z_i\overset{\mathrm{iid}}{\sim}\mathrm{Uniform}[q].
\]

All rules remain useful. This is the positive control.

Predictions:

- relevant posterior information grows with successful queries;
- learned repertoire and reward can grow linearly;
- \(B_{\rho_t}\) grows linearly for a suitable agent;
- fixed targets saturate.

### 9.2 Fixed-core redundant rules: RED

Let

\[
Z=(Z_1,\ldots,Z_d)
\]

be a fixed latent core and

\[
Y_i=f_i(Z)
\]

for public balanced functions \(f_i\).

The functions must be chosen so each surface label is marginally uniform, preventing trivial discrimination based on label frequency.

Two versions are required.

**RED-U: unrestricted additive redundancy.** With infinitely many rewarded derived rules and stochastic action channels, the frontier collapses. An oracle can reveal the finite core with probability \(d\), deploy \(M\) correct derived rules when revealed, and abstain otherwise. Setting \(d=\rho/(Mu)\) attains fixed expected reward while

\[
I(\Theta;A)\le dH(Z)\longrightarrow0
\]

as \(M\to\infty\). Therefore,

\[
B_\rho=0
\]

for every finite feasible \(\rho\). This “rare burst” result is a definitional expected-reward effect and an important adversarial control.

**RED-C: resource-controlled redundancy.** Bound or discount the total redundant reward by one declared stationary rule:

- a fixed maximum derived support;
- scoring each public equivalence class once;
- or fixed summable public weights.

Then, for feasible thresholds,

\[
B_\rho\le H(Z)=d\log q.
\]

The exact RED-C frontier, not latent rank alone, is the registered target. \(B_\rho=+\infty\) above its attainable reward.

Variants:

- exact copies;
- public permutations;
- Boolean or modular compositions;
- error-correcting-code-like redundant maps;
- and programs of increasing surface complexity generated from a fixed core.

### 9.3 Mixed-rank rules: MIX

Combine independent primitives with redundant derived rules:

\[
Y_i=
\begin{cases}
Z_i,&i\in\mathcal I,\\
f_i(Z_{\mathcal C_i}),&i\in\mathcal D.
\end{cases}
\]

The redundant component is resource-controlled so it cannot satisfy arbitrarily large reward targets by rare bursts. Its total reward contribution has a declared finite cap \(R_{\mathrm{red}}^{\max}\); asymptotic reward growth must come from independent primitives.

This prevents the estimator from merely learning to distinguish two extremes. Vary:

- effective latent rank;
- group size;
- dependency-graph topology;
- correlation strength;
- and the ratio of primitive to derived rules.

The estimator should recover the exact or certified reward-information frontier without being given the true factorization in held-out evaluation. “Effective rank” is a secondary correlate defined as the entropy of a minimal reward-sufficient latent statistic for the registered finite projection; it is not assumed equal to \(B_\rho\).

### 9.4 Aleatoric cosmetic novelty: ALEA

Append fresh random observations \(U_t\) that:

- are independent across time;
- do not persist in \(\Theta\);
- have no effect on reward.

They raise observation entropy, prediction error, and some novelty measures, but do not produce persistent environment information.

This condition is distinct from persistent trivia.

### 9.5 Persistent irrelevant trivia: TRIVIA

Add

\[
D_1,D_2,\ldots\overset{\mathrm{iid}}{\sim}\mathrm{Uniform}[q]
\]

to the static environment. Trivia coordinates are queryable through the same noisy channel as useful rules but never affect reward.

Predictions:

- a total-information agent can acquire trivia indefinitely;
- \(I((Z,D);H_t)\) can grow linearly;
- relevant information, reward, and \(B_{\rho_t}\) need not grow;
- the exact reward-information frontier is invariant to adding \(D\).

In fact, if \(D\perp Z\) and reward depends only on \(Z\),

\[
B_\rho^{(Z,D)}=B_\rho^Z.
\]

### 9.6 Public reward growth: PUBLIC

Three stationary variants are required.

**Public labels:** Some rewarded rules have publicly known labels, allowing rulebook size and reward to grow without environment information.

**Public bonus action:** The action includes a public integer \(k\), with reward

\[
r_\Theta(A,k)=r_\Theta(A)+g(k),
\]

where \(g\) is deterministic and known. An agent can increase \(k\) without learning \(\Theta\).

**PUBLIC-U:** If the public component alone can attain every finite reward threshold—for example, infinitely many publicly labeled rewarded rules or an unbounded \(g(k)\)—the total-reward frontier collapses exactly:

\[
B_\rho=0
\qquad
\text{for every finite feasible }\rho.
\]

**PUBLIC-C:** For the confirmatory factorial, use a fixed bounded public contribution \(0\le g(k)\le G_{\max}\) under a declared support/resource protocol. This shifts or truncates which hidden reward threshold is required without erasing the entire frontier.

For the canonical bounded-bonus version, \(k\) ranges over a finite public set containing \(k^\star\) with \(g(k^\star)=G_{\max}\). Because \(k^\star\) is independent of \(\Theta\),

\[
B_\rho^{\mathrm{PUBLIC-C}}
=
B_{\rho-G_{\max}}^{\mathrm{base}},
\]

using the convention that the base frontier is zero at thresholds attainable with zero information.

PUBLIC-U remains a paired degenerate control outside the main factorial. Report any bonus-adjusted hidden-component frontier as a separate metric, not as the total-reward bit-equivalent.

### 9.7 Finite target: FINITE

Only a fixed finite latent set affects reward. It supplies a bounded positive-information control:

\[
B_\rho\le N\log q.
\]

### 9.8 Independent opportunity anchor: OPP

This reproduces the conceptual infinite-armed control. The agent selects one opportunity rather than deploying a repertoire.

Because this changes action and reward semantics, it is a theoretical anchor, not a fully matched factorial cell.

The implementation must specify:

- the public candidate set accessible by time \(t\);
- whether only queried candidates may be selected;
- the payoff prior;
- and the support-size restriction.

No logarithmic claim may be inferred merely from an unrestricted countably infinite index. The finite observed-candidate bound must be explicit.

Because candidate exposure changes with horizon, OPP is analyzed as a finite-horizon family with a time-indexed frontier

\[
B_{\rho,t}^{\mathrm{opp}}.
\]

Its results are not pooled into the stationary Rulebook \(B_\rho\) analysis.

### 9.9 Ephemeral dynamic extension: EPH

Use \(W\) active slots initialized in stationarity. Each slot refreshes at a fixed hazard \(h\). Current reward depends only on current slot contents.

This is a stationary Markov process but not a static-\(\Theta\) bandit. It is labeled a dynamic-state extension.

Predictions:

- continual acquisition persists;
- total acquired information over the trajectory can grow;
- current decision information is bounded by the active-state entropy;
- catastrophic forgetting affects reward;
- current-performance bit-equivalent remains bounded in \(W\).

---

## 10. Factorial experimental design

### 10.1 Confirmatory stationary factorial

Baseline:

\[
q=4,\quad u=c=1,\quad \epsilon=0.1,\quad b=1.
\]

Factors:

| Factor | Confirmatory levels |
|---|---|
| Relevant structure | IND; RED-C |
| Irrelevant input comparison | none; ALEA; TRIVIA |
| Bounded public reward | absent; PUBLIC-C |
| Feedback | bounded-query P1 |
| Reward | additive |

This gives 12 primary environment cells before crossing agents. The three-level irrelevant-input comparison is heterogeneous: ALEA intervenes on fresh novelty, while TRIVIA intervenes on persistent learnable information. A separate \(2\times2\) robustness design crosses ALEA present/absent with TRIVIA present/absent.

RED-U and PUBLIC-U are paired adversarial controls, not crossed into cells where their zero-information collapse would erase the intended contrast.

### 10.2 Structural generalization matrix

- MIX with several effective ranks;
- hierarchical groups;
- sparse dependency graphs;
- dense public compositions;
- correlated sources with controlled spectrum;
- label permutations;
- redundant latent re-encodings;
- behaviorally equivalent action serializations.

At least one dependency family is held out from estimator development.

### 10.3 Feedback matrix

- exact bounded query;
- noisy bounded query;
- bounded semi-bandit;
- scalar aggregate Gaussian;
- scalar aggregate discrete;
- hybrid procedural feedback.

Feedback changes acquisition difficulty, not the environment’s decision frontier. Exact frontier artifacts are reused across feedback conditions with the same prior, action, and reward.

### 10.4 Reward matrix

- additive;
- positive affine transforms;
- logistic transform;
- hard clipping;
- normalized accuracy;
- coverage-adjusted accuracy;
- support cost;
- asymmetric false-deployment cost;
- public bonus.

Positive affine transforms satisfy the exact invariant

\[
B^{ar+b}_{a\rho+b}=B^r_\rho,
\qquad a>0.
\]

Here \(b\) is one constant added to the total scalar reward, independent of \(\Theta\) and \(A\). A per-coordinate offset \(b\|A\|_0\) changes deployment incentives and is a different reward function. The total-scalar identity is an implementation test, not an empirical hypothesis.

Nonlinear transformations define genuinely different reward semantics and may change the scaling class. In particular:

- logistic rewards can preserve a linear class when the transformed frontier’s divergence near its unattained supremum and the agent’s gap-to-supremum jointly imply \(B_{\rho_t}=\Theta(t)\);
- hard clipping can destroy open-endedness once extra capability no longer changes reward;
- normalization can remove the accumulation signal.

These outcomes are sensitivity results, not estimator failures by themselves.

Let \(S_\theta(A)\) denote the base whole-action additive score. Every configured reward locks an exact stationary formula:

\[
r^{\mathrm{aff}}_\theta(A)=aS_\theta(A)+b,
\qquad a>0;
\]

\[
r^{\mathrm{log}}_\theta(A)
=
\left(1+\exp\{-\gamma(S_\theta(A)-b_0)\}\right)^{-1};
\]

\[
r^{\mathrm{clip}}_\theta(A)
=
\min\{U,\max\{L,S_\theta(A)\}\};
\]

\[
r^{\mathrm{cost}}_\theta(A)
=
S_\theta(A)-\lambda\|A\|_0.
\]

For normalized accuracy, define reward as zero for empty support and otherwise use a declared function of correct and incorrect counts divided by \(\|A\|_0\). Coverage-adjusted reward adds a declared stationary function \(g(\|A\|_0)\); it never uses an “eligible prefix at time \(t\).” Each spec records attainable range, empty-support behavior, and whether transformation occurs before or after observation noise.

### 10.5 Action-representation matrix

- canonical explicit table;
- compact public program;
- decision diagram;
- finite automaton;
- neural policy;
- deliberately redundant serialization.

The symbolic representations compile to a canonical behavior before reward or mutual information is computed.

### 10.6 Registered matched-trajectory experiments

The factorial is supplemented by decisive matched comparisons. Matching schedules are developed on pilot seeds, frozen, and then evaluated on confirmatory seeds.

**Reward-matched**

Match pooled reward trajectories across:

- IND and RED-C by controlling deployment support and confidence;
- IND and PUBLIC-C by controlling the public bonus or public-label support;
- cumulative and EPH variants over a prespecified horizon.

The scientific contrast is bit-equivalent at equivalent raw performance.

**Total-information-matched**

Use scripted query allocations to match

\[
I(\Theta;H_t)
\]

between IND and TRIVIA while changing the relevant share of that information.

The scientific contrast is reward and \(B_{\rho_t}\) at equivalent acquired information.

**Novelty-matched**

Tune ALEA entropy or prediction difficulty so observation prediction error, compression improvement, or another prespecified novelty statistic matches an IND condition.

The scientific contrast is persistent relevant information and bit-equivalent at equivalent novelty.

**Task-count- and surface-complexity-matched**

Expose equal numbers of indexed rules and matched rule-description lengths in IND, RED, and MIX.

The scientific contrast is effective reward-relevant rank rather than task volume.

**Reliability/lottery-matched**

Match expected reward between:

- a reliable moderate-support deployment;
- and a rare high-support “jackpot” deployment.

Compare reward variance, lower quantiles, and \(I(\Theta;A)\). The primary construct remains minimum information for expected performance; a risk- or quantile-constrained frontier is reported as a separate secondary construct.

Matching is verified with equivalence tests using tolerances frozen after the pilot. If a match fails on confirmatory seeds, report the residual mismatch and use regression adjustment only as a sensitivity analysis; do not redefine the tolerance after seeing outcomes.

---

## 11. Frontier computation stack

Every reported frontier point is labeled as one of:

- exact;
- certified interval;
- achievable upper bound;
- converse lower bound;
- or heuristic.

### 11.1 Analytic solver

Implements Sections 4 and 5.

Outputs:

- \(p^\star\);
- \(r^\star\);
- \(\kappa\);
- \(B_1(r)\);
- \(B_N(\rho)\);
- \(B_\infty(\rho)\);
- analytic derivatives;
- and feasibility bounds.

### 11.2 Finite direct-channel solver

For finite \(\Theta\) and action set \(\mathcal A\), solve

\[
\min_{P(A\mid\Theta)} I(\Theta;A)
\]

subject to

\[
\mathbb E[r_\Theta(A)]\ge\rho,
\]

\[
P(A\mid\theta)\ge0,
\qquad
\sum_aP(a\mid\theta)=1.
\]

Use:

- a constrained convex solver;
- a Lagrangian Blahut-Arimoto-style solver;
- independent direct mutual-information calculation;
- and primal/dual certificates.

The Lagrangian stationary channel has the form

\[
P_\beta(a\mid\theta)
\propto
P_\beta(a)\exp(\beta r_\theta(a)).
\]

Sweeping \(\beta\) traces exposed frontier points. Constrained solves fill gaps and validate inversion.

### 11.3 Tensorized solver

The tensorized solver may run only after mechanically checking:

- independent source coordinates;
- additive reward;
- product-compatible action semantics;
- identical coordinate priors or a declared heterogeneous convolution;
- and no global support constraint that breaks separability.

It is regression-tested against full enumeration for small \(N\).

### 11.4 Direct behavioral channel estimator

For large action spaces, parameterize

\[
q_\phi(A\mid\Theta)
\]

directly over canonical actions.

Every learned problem declares a finite latent projection, maximum index, maximum support, and stopping semantics. Only the analytic stack may claim the true infinite frontier; learned finite-\(N\) results include an \(N\)-convergence study.

For rulebooks, use an autoregressive distribution over:

- sorted unique indices;
- predicted labels;
- and a stop decision.

Given a learned reference marginal \(m_\psi(A)\),

\[
\mathbb E_\Theta
D_{\mathrm{KL}}\!\left(
q_\phi(A\mid\Theta)\,\|\,m_\psi(A)
\right)
\]

is an upper bound on \(I(\Theta;A)\), because it equals

\[
I(\Theta;A)
+
D_{\mathrm{KL}}(q_\phi(A)\|m_\psi(A)).
\]

Together with achieved reward, this supplies an explicit feasible upper bound on \(B_\rho\).

With Monte Carlo evaluation, call the witness an achievable statistical upper bound only when:

- a lower confidence bound on its reward clears the target;
- an upper confidence bound on reference KL is used;
- \(m_\psi\) has support everywhere \(q_\phi\) does;
- and the sampled channel and seeds are retained.

Otherwise label the point heuristic.

### 11.5 Latent bottleneck estimator

The encoder-decoder model

\[
\Theta\to Z\to A
\]

is retained as an ambitious secondary estimator.

It reports:

- \(I(\Theta;Z)\) upper bound;
- induced-action reward;
- estimated or bounded \(I(\Theta;A)\);
- decoder compression gap;
- optimization variability;
- and disagreement with the direct action estimator.

No latent-bottleneck curve is labeled exact.

### 11.6 Converse bounds

Use the strongest available lower bound:

- exact convex dual;
- analytic tensorization;
- entropy caps;
- Fano-style error bounds;
- Donsker-Varadhan reward-deficit bounds;
- task-specific effective-rank bounds;
- and monotonicity or data-processing arguments.

The reported object is

\[
\underline B(\rho)
\le
B_\rho
\le
\overline B(\rho).
\]

Scaling is not classified when the lower and upper bounds support incompatible classes.

Certified deterministic envelopes and calibrated statistical intervals are different artifact types. A certified primal/dual envelope must contain the exact value at every checked grid point up to declared numerical-verification tolerance. Coverage percentages apply only to statistical or learned intervals.

Monotone/convex numerical repair cannot silently change a bound:

- lower bounds may use only a valid convex minorant;
- upper bounds require a feasible convex majorant backed by witnesses;
- visualization-only smoothing is never used for inference.

### 11.7 Frontier inversion

Given reward uncertainty

\[
\rho_t\in[\rho_t^-,\rho_t^+],
\]

report the conservative bit-equivalent interval

\[
\left[
\underline B(\rho_t^-),
\overline B(\rho_t^+)
\right].
\]

Inversion error is not summarized by a symmetric number where the frontier slope is nearly zero or vertical. Use censored or one-sided intervals.

---

## 12. Estimator calibration program

### 12.1 Calibration families

Include:

- \(q=2\) enumerable cases through the largest tractable \(N\);
- \(q=4\), \(N\le4\) full enumeration;
- larger analytic tensorized cases;
- finite-core redundancy;
- mixed-rank graphs;
- public reward;
- ALEA and TRIVIA augmentation;
- nonlinear rewards;
- and behaviorally equivalent action representations.

### 12.2 Development and held-out split

Development configurations may be used to:

- tune architectures;
- choose optimizer settings;
- determine reward grids;
- and set numerical tolerances.

Held-out calibration varies:

- \(q\);
- \(c/u\);
- \(N\);
- dependency graph;
- reward threshold;
- source-label permutation;
- action serialization;
- and distractor dimension.

No estimator is tuned after viewing confirmatory held-out results.

### 12.3 Initial numerical gates

Freeze final tolerances after a pilot. Initial targets are:

- one-coordinate primal/dual gap below \(10^{-8}\);
- enumerable multi-coordinate gap below \(10^{-6}\);
- analytic versus enumerated frontier error below \(10^{-8}\) for one coordinate;
- tensorized versus enumerated error below \(10^{-6}\) for \(N\le4\);
- normalized learned-frontier reward error at most \(0.02\);
- bit-equivalent inversion error at most \(\max(0.05\text{ nats},5\%)\) away from ill-conditioned slopes;
- calibrated statistical interval coverage at least \(95\%\) of prespecified exact grid points, with binomial uncertainty reported;
- certified intervals contain every checked exact grid point within numerical tolerance;
- distractor leakage below the larger of \(0.02\) nats and the frozen calibration-derived scale-normalized margin;
- and correct held-out ordering of bounded, logarithmic, sublinear, and linear controls.

Failure blocks interpretation of approximate large-instance results. It does not invalidate exact symbolic results.

### 12.4 Invariance canaries

The exact and approximate estimators are continuously tested against:

- irrelevant latent augmentation;
- source-label permutations;
- action-label permutations;
- redundant latent encodings;
- action reserialization;
- feedback-noise changes;
- positive affine reward transforms;
- and duplicate/no-op action fields.

---

## 13. Agents

All agents implement separate acquisition and deployment methods and declare structural privileges.

### 13.1 Capability manifest

Each agent declares whether it knows:

- the relevant/distractor mask;
- the coordinate factorization;
- the latent dependency graph;
- the target hierarchy;
- the true posterior family;
- the exact frontier;
- an approximate frontier;
- and the reward parameters.

Plots and tables must display or group by these privileges.

### 13.2 Random/uninformed

- random queries;
- abstaining deployment;
- optional deliberately random deployment diagnostic.

The primary uninformed policy should attain zero reward by abstaining.

### 13.3 Analytic sanity agent

The known-rate fresh-coordinate agent from Section 8.

### 13.4 Bayes reward-directed

Queries coordinates by expected decision value and deploys coordinate \(i\) only when:

\[
\max_jP(Y_i=j\mid H_t)>\tau.
\]

It predicts the posterior MAP label.

### 13.5 Fixed target

Learns only a declared target of size \(m\). Sweep:

\[
m\in\{8,32,128\}
\]

plus size-appropriate variants.

### 13.6 Prescheduled expansion

Use:

- too-slow;
- theoretically appropriate;
- too-fast;
- and extreme expansion schedules.

Schedules are defined exactly as functions of interaction count, not environment growth.

### 13.7 Unrestricted/rapid target

This baseline must be operationally defined. Examples:

- uniform query allocation over the current target;
- Thompson allocation over a rapidly growing target;
- posterior value-of-information allocation over all exposed indices.

The expected failure cannot be asserted merely because a target is large; a competent value-directed learner may ignore low-value dimensions.

### 13.8 Novelty-directed

Separate variants:

- prediction-error novelty;
- count-based novelty;
- compression improvement;
- latent-state visitation novelty;
- and procedural representation novelty.

ALEA should attract prediction-error novelty. TRIVIA should attract learnable novelty and persistent information objectives.

### 13.9 Total-information-gain

Selects queries maximizing expected reduction in full posterior entropy, including irrelevant persistent trivia.

To make H5 identifiable when useful and trivia queries have equal entropy reduction, each decision exposes one useful candidate and a declared number \(m_D\) of trivia candidates. Ties are broken uniformly over the exposed candidates. Sweep \(m_D\) and include an exclusive-query setting where choosing trivia consumes the useful query opportunity.

### 13.10 Relevant-information oracle

Maximizes information gain only about reward-relevant coordinates. This privileged diagnostic isolates objective mismatch from inference failure.

### 13.11 Ensemble frontier oracle

For current target \(\chi_m\), the population quantities

\[
C_{m,t}=I(\Theta_{\chi_m};H_t)
\]

and pooled target reward can be estimated across an ensemble of matched runs. An offline meta-controller may choose the next epoch’s shared schedule using the target frontier \(R_m^\star(C)\).

This is an ensemble-level oracle schedule, not an online single-trajectory agent. It knows the hierarchy and frontier and is labeled accordingly.

### 13.12 Estimated frontier-following curriculum

The runnable online curriculum uses quantities available from its current posterior. For target \(\chi_m\), compute:

- realized posterior KL to the target prior;
- current posterior-expected deployment value;
- and residual posterior value of perfect target information,
  \[
  G_m(h_t)
  =
  \mathbb E\!\left[
  \max_{a\in\mathcal A_m}r_\Theta(a)
  \mid h_t
  \right]
  -
  \max_{a\in\mathcal A_m}
  \mathbb E[r_\Theta(a)\mid h_t].
  \]

Expand only when:

- the upper confidence bound on \(G_m(h_t)\) is below threshold;
- the next target has positive conservative marginal reward per bit;
- and the estimator’s bound width is below a declared tolerance.

The realized posterior KL is a run-level diagnostic or decision feature, not the population mutual information \(I(\Theta;H_t)\).

### 13.13 Target-discovery curriculum

This ambitious version is not given the true relevance mask or factorization.

It maintains candidate target groups and estimates:

\[
\frac{
\text{conservative marginal reward}
}{
\text{additional target information}
}.
\]

Candidate generation can use:

- posterior dependency graphs;
- learned causal relevance;
- sparse group discovery;
- option or skill discovery;
- and procedural composition structure.

Performance is separated into:

- target discovery quality;
- information acquisition quality;
- and information utilization quality.

---

## 14. Posterior and information accounting

### 14.1 Exact categorical posterior

For IND and independent TRIVIA under q-ary symmetric observations, store per-coordinate observation counts and compute:

\[
P(\Theta_i=j\mid H_t)
\propto
P(\Theta_i=j)
\prod_{s:Q_s=i}P(O_s\mid\Theta_i=j).
\]

Untouched independent coordinates remain exactly at the prior and require no storage.

RED and MIX instead update the posterior over the minimal latent core using the likelihood of the queried surface map \(f_i(\Theta)\).

### 14.2 Information ledger

For independent primitive latents under factorized P1 feedback, a touched coordinate contributes:

\[
I_i(t)
=
\log q
-
\mathbb E\!\left[
H(P(\Theta_i\mid H_t))
\right].
\]

The coordinatewise sum equals total information only when the posterior factorizes over unique independent primitive latents. RED and MIX observations can couple core variables, and summing surface-rule marginal information would double-count shared knowledge.

The general implementation therefore computes

\[
\mathbb E_{H_t}
D_{\mathrm{KL}}\!\left(
P(\Theta_{\mathcal J_t}\mid H_t)
\;\|\;
P(\Theta_{\mathcal J_t})
\right)
\]

over the unique finite latent closure \(\mathcal J_t\), or uses an exact chain-rule conditional decomposition in a fixed primitive-latent order.

For one realized history \(h_t\), the software records Bayesian surprise

\[
K_t(h_t)
=
D_{\mathrm{KL}}\!\left(
P(\Theta_{\mathcal J_t}\mid h_t)
\;\|\;
P(\Theta_{\mathcal J_t})
\right).
\]

Only its expectation across the history-generating ensemble equals \(I(\Theta;H_t)\). Run-level KL, pooled mutual information, pooled \(B_{\rho_t}\), and ratios derived from them are stored as distinct types and are never substituted for one another.

Maintain distinct fields:

- reward-relevant primitive information;
- redundant-core information;
- persistent distractor information;
- dynamic-state information;
- and approximation residual.

The decomposition must reconcile with total KL within tolerance.

### 14.3 Aggregate posterior

Aggregate feedback couples coordinates. Implement:

- exact joint posterior for tiny cases;
- particle posterior;
- structured variational posterior;
- and amortized posterior for procedural cases.

Approximation error is evaluated on projections with exact solutions.

---

## 15. Metrics

### 15.1 Performance

- pooled expected reward;
- cumulative reward;
- reward variance and lower quantiles;
- deployment support;
- correct deployments;
- incorrect deployments;
- abstentions;
- mastered independent rules;
- effective mastered rank;
- transfer;
- and held-out compositional accuracy.

For symbolic rules, “mastered” means the frozen deployment predicts the correct surface label with posterior probability at least a predeclared threshold, initially \(0.95\), and passes the corresponding held-out correctness check. For procedural rules, use a frozen held-out error threshold selected before confirmatory runs.

For MIX, “effective rank” is not an informal graph statistic. Report separately:

- entropy of the minimal declared reward-sufficient latent statistic on the finite projection;
- algebraic rank where defined;
- and exact or certified \(B_\rho\).

### 15.2 Information

- total \(I(\Theta;H_t)\);
- relevant information;
- distractor information;
- dynamic-state information;
- \(I(\Theta;A_t)\) where estimable;
- \(B_{\rho_t}\) interval;
- and \(\overline B_T\) interval.

### 15.3 Novelty

- observation prediction error;
- compression improvement;
- count novelty;
- latent visitation;
- and behavioral novelty.

ALEA and TRIVIA are reported separately because fresh randomness and learnable persistent novelty are different constructs.

### 15.4 Useful-information efficiency

\[
\eta_t
=
\frac{B_{\rho_t}}{I(\Theta;H_t)}
\]

when the denominator is finite and positive.

Under exact aligned quantities,

\[
0\le\eta_t\le1
\]

by data processing when

\[
\Theta\to H_t\to A_t
\]

is the complete deployment-generation Markov chain. \(H_t\) must include every \(\Theta\)-dependent input used to produce the deployment, including observations, inherited model weights or memory, side information, and any evaluation context correlated with \(\Theta\). A violation can indicate omitted information as well as a numerical or solver bug.

For EPH, compute the ratio using current-state-aligned information and frontiers rather than the static-\(\Theta\) ledger.

This is a utilization diagnostic, not a claim that the agent stores exactly \(B_{\rho_t}\).

### 15.5 Frontier regret

\[
\Delta_t^{\mathrm{frontier}}
=
R^\star(C_t)-\rho_t.
\]

Report both:

- full-information frontier regret using \(C_t=I(\Theta;H_t)\);
- and relevant-information frontier regret using the appropriate reward-relevant projection.

The first penalizes distractor acquisition through inefficient use; the second isolates decision use conditional on relevant knowledge.

### 15.6 Compute and acquisition cost

- queries;
- environment steps;
- wall time;
- CPU/GPU time;
- peak memory;
- posterior-update time;
- frontier-solver time;
- and deployment-evaluation cost.

Information requirements and acquisition difficulty remain separate axes.

---

## 16. Statistical protocol

### 16.1 Replication unit

The full environment realization is the primary unit of replication.

Pair agents and matched conditions on:

- relevant latent seed;
- persistent distractor seed;
- aleatoric-noise tape where appropriate;
- query-observation noise tape;
- and frozen evaluation bank.

Noise is keyed by semantic coordinates such as

```text
(environment_seed, round, rule_index, query_ordinal, channel)
```

rather than sequential RNG position, so adaptive agents that make different queries do not shift one another’s noise tapes.

Use multiple algorithm seeds and declare whether they are crossed with or nested within environment seeds.

The exact analytic frontier is a deterministic population object and has no environment-seed variance. Environment and algorithm seeds quantify agent-performance variation and Monte Carlo approximation of pooled quantities. When the same algorithm seeds are crossed with several environment seeds, use crossed rather than automatically nested random effects.

### 16.2 Checkpoint weighting

In the symbolic benchmark, record deployment and expected reward every round.

For expensive neural evaluation, use geometric checkpoints but estimate

\[
\frac1T\sum_{t<T}B_{\rho_t}
\]

with interval weights or numerical integration over elapsed rounds. An unweighted mean of geometric checkpoints is invalid.

Report sensitivity to interpolation and checkpoint density.

### 16.3 Confidence intervals

Cluster bootstrap or hierarchical models at the environment-seed level. Checkpoints from one run are not independent replicates.

Track separately:

1. exact-solver primal/dual error;
2. approximate-estimator optimization variation;
3. reward-evaluation Monte Carlo error;
4. environment-seed variation;
5. algorithm-seed variation.

Aggregate realized posterior KL across matched runs to estimate population mutual information, and verify this identity by Monte Carlo:

\[
I(\Theta;H_t)
=
\mathbb E_{H_t}[K_t(H_t)].
\]

Do not divide a pooled \(B_{\rho_t}\) by one seed’s posterior KL or average nonlinear per-seed efficiency ratios as a substitute for the pooled estimand.

### 16.4 Confirmatory contrasts

Exact semantic invariants are regression tests, not statistical hypotheses:

- ALEA is absent from the frontier problem;
- exact TRIVIA augmentation leaves the frontier identical;
- feedback-only changes reuse the same frontier artifact;
- exact symbolic action reserialization canonicalizes identically;
- redundant latent re-encoding leaves the exact behavioral problem unchanged;
- positive affine total-reward remapping matches exactly.

Use equivalence tests for approximate estimators and empirically measured agent outcomes:

- learned-estimator distractor leakage is below a calibration-derived margin;
- approximate neural action representations agree within calibrated tolerance;
- feedback-noise conditions have equivalent independently solved frontiers while learning speeds differ;
- and registered matched trajectories satisfy their frozen matching tolerances.

Use superiority tests for:

- IND versus RED late-horizon bit-equivalent;
- relevant-information versus total-information acquisition efficiency;
- expanding versus fixed targets;
- and adaptive versus mistuned schedules.

Use simultaneous intervals or Holm correction for the small predeclared confirmatory family. Exploratory matrices emphasize effect sizes and uncertainty.

### 16.5 Seed count

Run a pilot to estimate between-environment and between-agent variation. Determine confirmatory seed count by power simulation for the smallest scientifically meaningful contrast, then freeze it.

---

## 17. Scaling inference

Finite experiments cannot prove asymptotic open-endedness. They can test whether observations agree with known scaling controls and whether a regime extrapolates to held-out horizons.

### 17.1 Candidate models

Fit:

- bounded saturation \(L-c\exp(-T/\tau)\), with \(c,\tau>0\);
- \(a+b\log(1+T)\), with \(b\ge0\);
- \(a+bT^\alpha\);
- linear \(a+bT\), with \(b\ge0\);
- and theory-specific analytic forms.

Parameter bounds and treatment of additive intercepts are frozen before confirmatory fitting. Zero-valued early checkpoints are fitted on the original scale and excluded only from log-slope diagnostics, never silently dropped from the trajectory.

### 17.2 Validation

- fit on the first \(70\%\) of elapsed horizon and reserve the final \(30\%\) for prediction;
- report predictive log likelihood or error;
- calculate dyadic local slopes;
- report \(\overline B_T/T\);
- report \(\overline B_T/\log(1+T)\);
- and bootstrap model comparisons by environment seed.

### 17.3 Linear-evidence rule

A reported linear regime requires:

- stable late-horizon doubling increments;
- late local-slope equivalence within the initial band \([0.9,1.1]\), frozen after pilot calibration;
- exclusion of the scientifically meaningful sublinear boundary \(\alpha\le0.8\);
- better held-out prediction than bounded and logarithmic alternatives;
- no systematic downward curvature over the longest horizons;
- and compatible scaling from lower and upper frontier bounds.

Repeated checkpoints use a hierarchical trajectory model or cluster bootstrap that preserves within-run covariance. The exact equivalence and exclusion bands, model parameter bounds, and predictive-error margin are frozen before confirmatory runs.

If the interval remains ambiguous, report partial identification rather than assign a class.

### 17.4 Scaling-classifier calibration

Build finite-prefix families with deliberately:

- bounded;
- logarithmic;
- \(T^{1/2}\);
- \(T^{2/3}\);
- and linear effective rank.

Use them to calibrate finite-horizon classification. They are calibration ladders, not claims that a time-varying prefix is one stationary open-ended environment.

---

## 18. Target-expansion phase diagram

Axes:

- target growth rate;
- query capacity \(b\);
- feedback noise \(\epsilon\);
- strict guessing margin;
- posterior approximation quality;
- and known versus discovered relevance.

Agents:

- fixed;
- slow;
- tuned;
- fast;
- extreme/unrestricted;
- frontier-gap adaptive;
- marginal-value-per-bit adaptive;
- and target-discovery adaptive.

Outcomes:

- reward;
- \(B_{\rho_t}\);
- \(\overline B_T\);
- target size;
- relevant information;
- distractor information;
- frontier regret;
- and compute.

The phase diagram should reveal:

- a saturation region;
- a learnable expansion region;
- an overexpansion/underutilization region;
- and any region where a competent value-directed learner avoids the expected failure.

Unexpected success by a rapidly expanding learner is a result, not an implementation problem.

The confirmatory adaptive claim is noninferiority to an oracle-tuned schedule under a margin frozen after pilot calibration, followed by robustness comparisons on held-out environment settings where the oracle schedule was not tuned. Superiority to an intentionally mistuned schedule is secondary evidence only.

---

## 19. Dynamic ephemeral extension

### 19.1 Model

There are \(W\) active slots. Each slot has a \(q\)-ary latent state and refreshes independently with hazard \(h\). Initialize the Markov chain in equilibrium.

The deployment action predicts current slot values. Old slot states have no current reward value after refresh.

Let \(X_t\) be the current active state and \(C_t\) any public context. Define the extension’s current-state frontier under the stationary marginal:

\[
B_\rho^{\mathrm{state}}
=
\inf_{P(A\mid X_t,C_t):\,
\mathbb E[r(X_t,A,C_t)]\ge\rho}
I(X_t;A\mid C_t).
\]

Current useful-information efficiency compares this frontier with \(I(X_t;H_t\mid C_t)\), not information about the full trajectory or the static Rulebook latent.

### 19.2 Questions

- Can total trajectory information grow while current decision information stays bounded?
- How do memory size and forgetting affect reward without changing the current frontier?
- Does a learner incorrectly classified by cumulative information appear open-ended?
- Does the bit-equivalent remain stable under continual turnover?

### 19.3 Reporting boundary

This experiment is an extension of the diagnostic concept beyond the source paper’s static bandit setting. It must not be used as direct evidence for or against the original formal definition without stating the generalization.

---

## 20. Procedural compositional extension

### 20.1 World

Each primitive name denotes one hidden transformation selected from a finite \(q\)-ary family. A task supplies:

- a public input object;
- a sequence or graph of primitive names;
- and a requested output.

The correct output depends on the latent primitive identities.

### 20.2 Conditions

At minimum:

- independent primitives;
- fixed-core derived primitives;
- mixed-rank primitives;
- ALEA rendering distractors;
- persistent irrelevant trivia;
- cumulative versus ephemeral evaluation;
- and public reward shortcuts.

### 20.3 Evaluation

Use:

- held-out inputs;
- held-out compositions;
- increased composition depth;
- novel recombinations;
- and adversarial distractor shifts.

The deployed policy cannot query hidden rule feedback during evaluation. Runtime observations may describe the public task but may not reveal \(\Theta\) for free.

### 20.4 Behavioral action approximation

Where the finite task domain is exhaustive, identify policies by their full input-output behavior.

Otherwise:

- use frozen diagnostic task banks of increasing size;
- measure collision and disagreement rates;
- repeat estimates across independent banks;
- and report the behavioral-quotient approximation error.

A sampled policy fingerprint is not treated as exact equivalence. Collisions merge distinct actions and can bias estimated action information downward.

Use four disjoint bank families:

- behavioral fingerprint construction;
- frontier fitting;
- recurring reward evaluation;
- and final untouched validation.

The final bank cannot affect training, hyperparameter choice, early stopping, or estimator selection.

### 20.5 Neural models

Include:

- small recurrent or transformer policy;
- explicit memory model;
- modular tool model;
- and compact program synthesizer where feasible.

Separate:

- representation learning;
- target discovery;
- acquisition;
- retention;
- and compositional execution.

---

## 21. Software architecture

```text
information-paper/
  pyproject.toml
  README.md
  configs/
    environment/
    feedback/
    reward/
    agent/
    estimator/
    experiment/
    sweeps/
  src/rulebook/
    core/
      types.py
      protocols.py
      rng.py
      behavior.py
      rewards.py
    environments/
      independent.py
      redundant.py
      mixed_rank.py
      distractors.py
      public_reward.py
      opportunities.py
      ephemeral.py
      procedural.py
    feedback/
      query.py
      semi_bandit.py
      aggregate.py
      channels.py
    posteriors/
      categorical_product.py
      finite_joint.py
      latent_core.py
      composite.py
      dynamic_slots.py
      particle.py
      variational.py
      information_ledger.py
    frontier/
      problem.py
      one_coordinate.py
      tensorized.py
      exact_channel.py
      blahut_arimoto.py
      dual.py
      direct_behavioral.py
      latent_bottleneck.py
      bounds.py
      inversion.py
      calibration.py
      cache.py
    agents/
      random.py
      sanity.py
      bayes_reward.py
      fixed_target.py
      scheduled_target.py
      unrestricted.py
      novelty.py
      information_gain.py
      frontier_following.py
      target_discovery.py
    evaluation/
      checkpoints.py
      reward.py
      information.py
      novelty.py
      scaling.py
      uncertainty.py
    orchestration/
      run.py
      sweep.py
      seeds.py
      manifests.py
      artifacts.py
  tests/
    unit/
    property/
    solver/
    integration/
    regression/
  experiments/
    phase0_math/
    phase1_solvers/
    phase2_calibration/
    phase3_construct_validity/
    phase4_agents/
    phase5_distractors/
    phase6_expansion/
    phase7_rewards/
    phase8_dynamic/
    phase9_procedural/
  scripts/
    run_experiment.py
    build_figures.py
    validate_artifacts.py
```

---

## 22. Core software contracts

Illustrative interfaces:

```python
class Environment(Protocol):
    spec: EnvironmentSpec

    def reset(self, seed: Seed) -> EnvironmentState: ...

    def evaluate(
        self,
        latent: LatentProjection,
        action: DeploymentAction,
        context: EvaluationContext,
    ) -> RewardVector: ...

    def finite_problem(
        self,
        projection: ProjectionSpec,
    ) -> FrontierProblem: ...


class InteractionProtocol(Protocol):
    def step(
        self,
        environment: Environment,
        state: EnvironmentState,
        action: TrainAction,
    ) -> tuple[EnvironmentState, TrainOutcome]: ...


class Agent(Protocol):
    capabilities: AgentCapabilities

    def reset(self, prior: Prior, seed: Seed) -> None: ...

    def select_train_action(
        self,
        context: TrainContext,
    ) -> TrainAction: ...

    def observe(
        self,
        action: TrainAction,
        outcome: TrainOutcome,
    ) -> None: ...

    def deployment_distribution(
        self,
        context: CheckpointContext,
    ) -> DeploymentDistribution: ...


class FrontierProvider(Protocol):
    def solve(
        self,
        problem: FrontierProblem,
    ) -> FrontierEnvelope: ...
```

`TrainAction` is a tagged union:

- `QueryAction` for P1;
- `DeploymentTrainAction` for P0;
- `SemiBanditTrainAction(deployment, feedback_subset)` for P2.

Checkpoint `DeploymentAction` remains separate and side-effect-free.

Required records:

**FrontierProblem**

- finite prior;
- canonical behavioral action space;
- reward matrix;
- units;
- feasibility range;
- structural assumptions;
- provenance hash.

**FrontierEnvelope**

- reward grid;
- lower information bound;
- upper information bound;
- feasible primal witnesses;
- dual certificates;
- solver diagnostics;
- raw curve;
- optional monotone-convex numerical repair, clearly distinguished from raw output.

**RunCheckpoint**

- realized reward samples;
- realized posterior KL;
- deployment witness and seed;
- run-local novelty, support, mastery, target size, and compute.

**CheckpointEstimate**

- pooled reward and interval;
- bit-equivalent interval;
- population relevant and distractor mutual information;
- population useful-information efficiency;
- aggregated novelty, support, and mastery;
- frontier regret;
- uncertainty decomposition;
- and semantic hashes.

---

## 23. Configuration and reproducibility

Use typed, versioned configuration files validated into immutable models.

Every run records:

- fully resolved config;
- code commit;
- dirty-tree hash including tracked diffs and untracked source/config content;
- dependency-lock hash;
- container or environment digest;
- solver, BLAS, numeric-precision, and deterministic-mode versions;
- CUDA and cuDNN versions where applicable;
- analysis-code hash;
- deterministic seed tree;
- environment semantic hash;
- reward semantic hash;
- action semantic hash;
- frontier artifact hash;
- hardware;
- wall time;
- and metric units.

One run cell is:

```text
environment × feedback × reward × agent × seed
    -> immutable training events
    -> side-effect-free checkpoints
    -> independently cached frontier
    -> immutable metric artifact
```

Frontiers are cached separately from agents because the frontier is an environment property.

Suggested artifact layout:

```text
artifacts/<experiment>/<run_hash>/
  manifest.json
  config.resolved.yaml
  events.jsonl.zst
  checkpoints.parquet
  deployments/
  posterior_summaries/
  frontier/
    curve.parquet
    witnesses/
    certificates/
    diagnostics.json
  metrics.parquet
  stderr.log
```

Figures read only validated immutable artifacts.

---

## 24. Test suite and acceptance gates

### 24.1 Semantic tests

- every lazily generated primitive latent and surface label is invariant to query order and process count;
- Checkpointing does not change training state or RNG.
- Abstention earns exactly zero.
- Uninformed deployment earns the declared negative margin.
- Duplicate indices cannot inflate reward.
- Equivalent behaviors receive identical reward.
- Feedback noise does not alter the frontier hash.
- All finite deployments have finite reward.
- Information ledgers never instantiate infinite entropy.

### 24.2 Mathematical solver tests

- channel rows are normalized and nonnegative;
- primal reward constraints hold;
- direct and solver MI calculations agree;
- one-coordinate analytic and exact solvers agree;
- tensorized and enumerated solvers agree for small \(N\);
- \(B_\rho\) is nondecreasing and convex;
- \(B_\rho=0\) through the zero-information optimum;
- \(B_\rho=+\infty\) above attainable reward;
- RED-U rare-burst witnesses drive information toward zero at fixed finite reward as support grows;
- RED-C never exceeds \(H(Z)\) on feasible thresholds;
- MIX cannot attain asymptotically growing target reward solely from its capped redundant component;
- TRIVIA leaves the exact frontier unchanged;
- affine reward transforms map exactly;
- the equality-margin adversarial case approaches zero information as \(N\) grows.

### 24.3 Integration tests

- the sanity agent matches analytic reward and bit-equivalent slopes;
- fixed targets saturate at the predicted bound;
- expanding targets pass the positive-control growth test;
- TRIVIA raises distractor information for the total-information agent;
- ALEA raises novelty without persistent information;
- feedback noise slows acquisition without changing the exact frontier;
- RED-U and RED-C match their distinct predicted frontiers;
- PUBLIC-U collapses and PUBLIC-C matches its bounded closed-form transformation;
- EPH matches its current-state frontier while acquiring trajectory information continually;
- seed-average realized posterior KL matches population mutual information within Monte Carlo error;
- correlated RED/MIX ledgers do not double-count;
- P0/P2 traces satisfy declared feedback-bandwidth or information reporting rules;
- evaluation, fingerprint, fitting, and final-validation banks are disjoint;
- every approximate upper-bound point retains a feasible channel witness and confidence-adjusted reward.

### 24.4 Reproducibility tests

- identical config and seed produce identical canonical scientific-payload hashes for deterministic symbolic runs;
- runtime metadata such as timestamps, wall time, hardware, and nondeterministic file metadata is excluded from the scientific-content hash;
- local and distributed symbolic sweeps are semantically equal;
- neural/GPU reruns agree within frozen numerical tolerances;
- interrupted runs resume without duplicated events;
- figure builds are deterministic;
- incompatible semantic hashes are rejected;
- every published result has complete provenance.

---

## 25. Workstreams and phase gates

The scope remains intact. Gates determine interpretation and ordering, not whether later work is permanently excluded.

### Workstream A: formal mathematics

Deliver:

- strict-margin collapse theorem;
- exact one-coordinate frontier;
- tensorization proof;
- infinite linear frontier;
- distractor invariance;
- unrestricted redundant rare-burst collapse and resource-controlled entropy cap;
- bounded-reward analyses;
- and acquisition-capacity upper bound.

Gate A:

- derivations reviewed;
- analytic implementation passes regression tests.

### Workstream B: simulator and posteriors

Deliver:

- stationary latent generator;
- P0/P1/P2 protocols;
- exact categorical posterior;
- information ledger;
- aggregate posterior approximations;
- and all symbolic environment variants.

Gate B:

- semantic and analytic end-to-end tests pass.

### Workstream C: exact frontier stack

Deliver:

- direct convex solver;
- BA-style solver;
- dual certificates;
- tensorized solver;
- inversion;
- and immutable frontier artifacts.

Gate C:

- exact tolerances pass on enumerable cases.

### Workstream D: approximate estimators

Deliver:

- direct behavioral channel estimator;
- latent bottleneck estimator;
- lower-bound estimators;
- calibration harness;
- and held-out evaluation.

Gate D:

- calibration coverage and leakage criteria pass.

If Gate D fails, large-instance conclusions use exact or analytic cases and report estimator failure as a central result.

### Workstream E: agents and construct validity

Deliver:

- all baseline agents;
- confirmatory factorial;
- distractor sweep;
- matched information/reward comparisons;
- and action invariance tests.

Gate E:

- predeclared confirmatory contrasts completed with frozen analysis.

### Workstream F: adaptive curricula

Deliver:

- oracle frontier-following;
- estimated frontier-following;
- marginal-value-per-bit curriculum;
- target discovery;
- and phase diagram.

Gate F:

- adaptive agents evaluated across multiple \(q\), margins, noise levels, and dependency structures.

### Workstream G: reward and representation stress tests

Deliver:

- affine invariant;
- logistic;
- clipping;
- normalization;
- support cost;
- table/program/policy comparisons;
- and risk-constrained secondary frontiers.

### Workstream H: dynamic extension

Deliver:

- Markov ephemeral environment;
- forgetting and memory-capacity agents;
- and current-versus-trajectory information analysis.

### Workstream I: procedural neural extension

Deliver:

- compositional world;
- neural and programmatic policies;
- behavioral fingerprint analysis;
- approximate frontiers;
- held-out depth and recombination tests;
- and target-discovery evaluation.

---

## 26. Predeclared success and falsification criteria

### 26.1 Construct validity

**Convergent validity**

- bit-equivalent matches exact or certified reward-information frontiers across controlled effective structures;
- it correlates with independently mastered useful rules;
- it predicts held-out composition performance when capability accumulation is required.

**Discriminant validity**

- it separates from raw reward;
- novelty;
- task count;
- total information;
- and continual turnover.

**Intervention validity**

- adding independent useful distinctions changes the frontier;
- adding irrelevant entropy does not;
- replacing independent rules with public compositions yields the predicted bounded or zero-information redundant frontier.

**Representation validity**

- behaviorally equivalent actions agree;
- redundant serialization does not change classification.

**Acquisition invariance**

- feedback noise changes learning speed but not the decision frontier.

**Predictive validity**

- calibrated estimators predict held-out sizes, structures, and horizons.

### 26.2 Strong falsifiers of the empirical measurement claim

- learned estimates rise with exact-irrelevant distractor entropy;
- action serialization changes the frontier materially;
- direct and latent estimators disagree beyond declared bounds;
- confidence intervals miss exact curves systematically;
- the scaling classifier fails on known controls;
- the analytic sanity agent fails to reproduce its known frontier and slope;
- or approximate bounds remain too wide to distinguish registered controls.

### 26.3 Honest interpretations of partial failure

- Exact benchmark succeeds, estimator fails: the construct remains coherent but not yet measurable at scale.
- Estimator succeeds, generic agents fail: metric validation succeeds; autonomous open-ended learning does not.
- Oracle adaptive curriculum succeeds, target discovery fails: schedule adaptation is viable under known structure.
- Procedural ordering fails: symbolic construct validity does not transfer cleanly to function approximation.
- Nonlinear rewards change the class: reward relativity is empirically consequential.
- Bit-equivalent grows under lottery policies without reliable mastery: the expected-performance construct is working as defined, but capability interpretation requires the risk-constrained secondary frontier.
- A generic learner saturates in analytically open-ended IND: this limits the learner, not the environment.
- Target discovery requires the hidden factorization: this weakens the autonomous algorithmic claim, not exact metric validity.

---

## 27. Planned figures

1. **Exact one-coordinate frontier** with analytic and numerical agreement.
2. **Tensorization test** comparing enumeration and \(N B_1(\rho/N)\).
3. **Central construct-validity panel:** reward, novelty, total information, relevant information, and bit-equivalent across IND, RED, ALEA, TRIVIA, and PUBLIC.
4. **Distractor decomposition:** fresh novelty versus persistent irrelevant information.
5. **Estimator calibration:** exact curve, achievable upper bound, converse lower bound, and uncertainty.
6. **Action representation invariance:** table, program, and neural behavioral quotient.
7. **Target-expansion phase diagram.**
8. **Adaptive curriculum comparison.**
9. **Scaling-model held-out prediction.**
10. **Reward semantics:** affine, logistic, clipped, and normalized.
11. **Dynamic turnover:** continual information versus bounded current bit-equivalent.
12. **Procedural transfer:** held-out composition depth and effective rank.

---

## 28. Paper narrative

1. Open-ended learning is often assessed through novelty, task production, reward, or information gain.
2. Controlled interventions can separate those quantities.
3. Bit-equivalent asks what environment information a behavioral decision minimally requires.
4. Measuring the infimum is difficult and cannot be replaced by hidden-state information.
5. The Infinite Rulebook supplies an exactly solvable stationary positive control.
6. A strict negative uninformed margin is necessary; otherwise the infinite frontier collapses.
7. Analytic and finite convex solvers establish ground truth.
8. Matched interventions separate independent useful structure, redundancy, aleatoric novelty, persistent trivia, and public reward.
9. Direct and latent estimators are evaluated as partially identified approximations.
10. Expanding and adaptive targets are tested against fixed, unrestricted, novelty, and information objectives.
11. Reward and action semantics expose real sensitivities.
12. Dynamic and procedural extensions test how far the diagnostic transfers.
13. The conclusion is limited to the evidence: bit-equivalent is useful where behavioral actions and reward-information frontiers can be identified with defensible bounds.

---

## 29. Immediate implementation order

1. Create the package skeleton and typed core records.
2. Implement the baseline \(q=4,u=c=1\) reward and canonical rulebook.
3. Implement \(J_q(p)\), \(B_1(r)\), \(B_N(\rho)\), and \(B_\infty(\rho)\).
4. Add strict-margin and equality-collapse tests.
5. Implement the lazy stationary latent generator.
6. Implement P1 bounded queries and the exact categorical posterior.
7. Implement side-effect-free checkpointing.
8. Implement the analytic sanity agent and verify its slope.
9. Implement the finite direct-channel solver and compare it with the analytic frontier.
10. Implement IND, RED, ALEA, TRIVIA, and PUBLIC.
11. Implement the information ledger and core metrics.
12. Add fixed, scheduled, novelty, total-information, and reward-directed agents.
13. Run the pilot needed to freeze tolerances and seed counts.
14. Freeze confirmatory configs.
15. Proceed in parallel on aggregate feedback, approximate estimators, adaptive curricula, reward stress tests, the dynamic extension, and the procedural wrapper.

---

## 30. Final positioning

The ambitious contribution is not merely a new benchmark or an implementation of truncated Thompson sampling. It is a measurement program with four linked products:

1. a mathematically characterized environment family;
2. a calibrated partial-identification stack for reward-relevant information;
3. a controlled factorial construct-validity study separating useful information from novelty and redundancy;
4. and an algorithmic study of how learning targets should expand when the useful frontier is not yet saturated.

The first implementation should make every later claim auditable. The exact symbolic frontier is the anchor; learned estimators, adaptive target discovery, dynamic turnover, and neural composition extend outward from that anchor without changing what has and has not been established.

---

## 31. References

1. Wanqiao Xu, Yifan Zhu, and Benjamin Van Roy. [*An Information-Theoretic Definition for Open-Ended Learning*](https://arxiv.org/abs/2606.08369). 2026.
2. Dilip Arumugam and Benjamin Van Roy. [*Deciding What to Learn: A Rate-Distortion Approach*](https://arxiv.org/abs/2101.06197). 2021.
3. Dilip Arumugam and Benjamin Van Roy. [*The Value of Information When Deciding What to Learn*](https://arxiv.org/abs/2110.13973). 2021.
4. Thomas M. Cover and Joy A. Thomas. *Elements of Information Theory*. Wiley.
