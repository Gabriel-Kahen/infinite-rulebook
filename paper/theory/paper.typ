#set document(
  title: "When Does Reward Require Information?",
  author: "Gabriel Kahen",
  keywords: (
    "open-ended learning",
    "rate-utility",
    "rate-distortion",
    "mutual information",
    "reward transformations",
  ),
)

#set page(
  paper: "us-letter",
  margin: (top: 0.78in, bottom: 0.78in, left: 0.88in, right: 0.88in),
  footer: context [
    #align(center)[#text(size: 8.5pt, fill: rgb("555555"))[#counter(page).display("1")]]
  ],
)
#set text(font: "Libertinus Serif", size: 10.4pt, lang: "en")
#set par(justify: true, leading: 0.61em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 14pt, weight: "bold", fill: rgb("173B57"))
#show heading.where(level: 2): set text(size: 11.5pt, weight: "bold", fill: rgb("244E6A"))
#show heading: set block(above: 1.15em, below: 0.55em)
#show link: set text(fill: rgb("1F5A7A"))
#set list(indent: 1.15em, body-indent: 0.55em, spacing: 0.35em)
#set enum(indent: 1.25em, body-indent: 0.55em, spacing: 0.35em)
#set table(inset: 5pt, stroke: 0.45pt + rgb("BBC5CC"))

#let navy = rgb("173B57")
#let blue = rgb("2D6382")
#let pale = rgb("F3F7FA")
#let warm = rgb("FBF8F1")
#let muted = rgb("5E6B73")

#let statement(kind, name, body) = block(
  breakable: true,
  width: 100%,
  stroke: (left: 1.8pt + blue),
  fill: pale,
  inset: (left: 10pt, right: 9pt, top: 7pt, bottom: 7pt),
  above: 0.75em,
  below: 0.75em,
)[
  #strong[#kind: #name.] #body
]

#let definition(name, body) = block(
  breakable: true,
  width: 100%,
  stroke: (left: 1.8pt + rgb("8D713B")),
  fill: warm,
  inset: (left: 10pt, right: 9pt, top: 7pt, bottom: 7pt),
  above: 0.75em,
  below: 0.75em,
)[
  #strong[Definition: #name.] #body
]

#let proofsketch(body) = block(
  breakable: true,
  width: 100%,
  inset: (left: 9pt, right: 4pt, top: 2pt, bottom: 4pt),
  stroke: (left: 0.7pt + rgb("A9B2B8")),
)[
  #emph[Proof sketch.] #body
]

#align(center)[
  #text(size: 21pt, weight: "bold", fill: navy)[When Does Reward Require Information?]
  #v(3pt)
  #text(size: 14pt, weight: "medium", fill: blue)[Foundations and Failure Modes of the Bit-Equivalent]
  #v(10pt)
  #text(size: 11.5pt)[Gabriel Kahen]
  #v(2pt)
  #text(size: 9.5pt, fill: muted)[August 2026 - Preprint]
]

#v(8pt)
#line(length: 100%, stroke: 0.8pt + rgb("9FAAB1"))
#v(8pt)

#block(
  width: 100%,
  fill: rgb("F7F9FA"),
  radius: 3pt,
  inset: 11pt,
  stroke: 0.5pt + rgb("D4DBDF"),
)[
  #align(center)[#strong[Abstract]]
  #v(3pt)
  The bit-equivalent of a reward level is the infimum mutual information between an unknown environment and an action needed to attain that reward in expectation. Xu, Zhu, and Van Roy recently use growth of this quantity to define open-ended learning. Mathematically, however, the static object is the classical rate-utility frontier. We ask when promoting that frontier to a semantic criterion yields a representation-robust and nondegenerate notion of capability relative to a declared contract. Positive-affine reward transformations conjugate every frontier and, within continuous strictly increasing pointwise maps and universal continuous threshold relabelings, are the unique such class. In contrast, we construct a deterministic bandit whose open-endedness classification is reversed by the invertible map $r mapsto r^3$ even though reward histories are isomorphic. We prove exact invariance under reward-sufficient source reductions and behaviorally equivalent action quotients that admit a source-independent lift. Bounded reward yields a Pinsker lower bound above the best uninformed reward; any vanishing-information sequence that maintains a fixed positive reward gap must lose uniform integrability under its induced and independence-reference laws, and unbounded leverage supplies explicit collapse witnesses. Finally, translating Shannon's finite product law into reward coordinates and passing to a countable iid product gives a local-price formula under an explicit finite-expected-support contract. The criterion is therefore cardinal rather than ordinal and depends on the declared reward, prior, behavioral representation, tail conditions, and admissible channels.
]

= Introduction

Open-ended learning is meant to describe a system whose useful capabilities can continue to grow through interaction. Reward growth alone is inadequate: a large scalar payoff may be available without learning, while difficult behavior may be compressed into a small reward range. Novelty and total information acquisition are also inadequate because they do not distinguish payoff-relevant information from distraction. Xu, Zhu, and Van Roy propose a direct information-theoretic alternative: price each expected reward level by the least information about the environment that a successful action must contain, and call an environment open-ended when an agent makes this price grow linearly on average @xu2026.

For a Bayesian source $Theta$, action $A$, and mean reward $r(Theta, A)$, the proposed price is

$
  B(rho) = inf_(P(A|Theta): E[r(Theta,A)] >= rho) I(Theta; A).
$

This is a compelling object. It suppresses information that is not needed for reward, and it asks about successful behavior rather than the entire interaction history. But the same formula is also familiar. Replacing distortion by a constant minus reward turns Shannon's rate-distortion function into $B$ @shannon1959; information-constrained decision theory explicitly calls the corresponding object a rate-utility frontier @genewein2015. The mathematical optimization is therefore not new. The distinct issue studied here is the semantic load placed on it. A coding frontier may legitimately change when the distortion function changes. A proposed criterion for open-ended capability should therefore state which changes of reward, source, action representation, and admissibility preserve its classification.

This paper develops a foundations layer for that use. Its organizing question is:

#align(center)[
  #block(width: 86%, fill: pale, radius: 4pt, inset: 10pt)[
    #emph[Under which explicit contracts is the bit-equivalent a representation-robust, nondegenerate price of reward-relevant capability?]
  ]
]

The answer has a positive side—exact invariances, noncollapse certificates, and composition laws—and a sharp negative side that fixes their scope.

Arbitrary increasing transformations are not safe. We give an explicit deterministic bandit that is open-ended under reward $r$ and non-open-ended under $r^3$, although cubing bijectively maps every reward history and leaves actions, timing, and nonreward observations unchanged. Unbounded rewards can also make a positive target cost zero information as an infimum: increasingly valuable informed behavior is activated with increasingly small probability. Even bounded reward can have a zero-valued but unattained frontier at the zero-information boundary when the action space is noncompact. Finally, task count alone does not determine countable information demand; the controlling quantity is the local slope of the one-component frontier at zero.

== Contributions

The paper's contribution is a foundations analysis of the new use of the frontier, not the introduction of the frontier itself. Concretely:

1. *Reward scale.* We record exact positive-affine conjugacy and formulate classical positive-affine uniqueness at the frontier level. We then construct an invertible monotone transformation that reverses the Xu-Zhu-Van Roy open-endedness classification while preserving the deterministic reward-history experiment.

2. *Representation.* We identify exact sufficient-source and behavioral-action quotients. The action theorem isolates the source-independent admissible lift required for equality and distinguishes raw serialization information from the optimized behavioral frontier.

3. *Nondegeneracy.* We derive a quantitative Pinsker certificate for bounded reward and a two-law uniform-integrability necessity theorem for positive-gap collapse. Explicit constructions separate bounded boundary nonattainment from rare-burst and increasing-leverage positive-gap collapse.

4. *Composition.* We translate Shannon's finite product-source theorem into reward coordinates. Under a pointwise null action and a finite-expected-support contract, we prove that the countable iid frontier linearizes at the one-component frontier's right slope at zero.

The main paper-specific result is the classification-reversal construction. Source/action reduction, bounded noncollapse, and finite composition are positioned as classical tools or exact specializations. The countable local-price law is positioned more cautiously as a delimited corollary with unresolved priority. We make their joint implications explicit for this criterion; failure to locate an antecedent is not treated as proof of priority.

== Results at a glance

#figure(
  table(
    columns: (1.08fr, 1.55fr, 2.15fr),
    align: (left, left, left),
    table.header(
      [#strong[Dimension]],
      [#strong[Safe contract]],
      [#strong[Failure outside the contract]],
    ),
    [Reward scale], [Global positive-affine map], [Cubing can reverse open-endedness despite invertible feedback],
    [Source], [Exact reward-sufficient statistic], [Conditional-mean aggregation need not preserve attainable reward],
    [Action], [Behavioral quotient with admissible source-independent lift], [Constrained or unliftable quotient can lower the frontier],
    [Tails and topology], [Bounded reward, or UI under both laws along a candidate sequence; compactness for boundary attainment], [Rare bursts collapse a positive gap; noncompactness defeats boundary attainment],
    [Composition], [Independent sources, product actions, additive reward], [Dependence, coupling, or hard support caps invalidate tensorization],
  ),
  caption: [The structural contracts established in this paper. "Safe" means equality or noncollapse under the stated theorem, not invariance under every change to the model.],
) <tab:summary>

== Scope

The object studied here is primarily a one-shot Bayesian decision frontier. We invoke a sequential policy only for the explicit classification reversal and use exactly the average-bit-equivalent convention of @xu2026. A statement about the frontier says what successful behavior must encode; it does not show that a sequential agent can discover that behavior. We do not propose an estimator, run a benchmark, or claim empirical validation. Standard-Borel extensions are stated only where the required regular conditional laws, measurable lifts, and integrability conditions are explicit.

= Bayesian reward-information frontiers

Let a finite Bayesian decision problem be

$
  cal(D) = (cal(T), mu, cal(A), r),
$

where $Theta tilde mu$ is the unknown source, $cal(A)$ is the action set, and $r: cal(T) times cal(A) -> RR$ is mean reward. A behavioral channel $K(a|theta)$ induces joint law $P_(Theta,A) = mu K$, reward

$ R_(cal(D))(K) = E_K[r(Theta,A)], $

and information cost

$ I_(cal(D))(K) = I_K(Theta; A). $

#definition("Reward-information frontier")[
  For threshold $rho in RR$,
  $
    B_(cal(D))(rho)
    = inf_(K: R_(cal(D))(K) >= rho) I_(cal(D))(K),
  $
  with the infimum of the empty set equal to $+infinity$. All logarithms are natural, so information is measured in nats even though we retain the established term "bit-equivalent."#footnote[The terminology follows @xu2026, who likewise computes information in nats.]
]

The best source-independent reward is

$
  R_0(cal(D))
  = sup_(nu in cal(P)(cal(A))) E_(mu times nu)[r(Theta,A)]
  = sup_(a in cal(A)) E_mu[r(Theta,a)].
$

The last equality follows from linearity in the action distribution. It is a maximum in a finite problem but may be an unattained supremum in a noncompact action space. We use standard information-theoretic notation @cover2006.

For finite alphabets, $B_(cal(D))$ is nondecreasing and convex on its feasible reward interval. Monotonicity follows from nested feasible sets. Convexity follows by mixing channels and using convexity of mutual information in the channel for fixed source law. Under the change of variables $d(theta,a)=C-r(theta,a)$ with $C>=max_(theta,a) r(theta,a)$, the reward constraint becomes a conventional nonnegative distortion constraint and $B$ is Shannon's rate-distortion function in reward coordinates @shannon1959. The best uninformed reward $R_0$ is the reward form of the classical zero-rate cutoff.

== Connection to open-endedness

In the bandit convention of @xu2026, an agent $pi$ induces expected rewards $rho_t = E_pi[R_(t+1)]$. Its average bit-equivalent through time $T$ is

$
  overline(B)_T(pi) = 1/T sum_(t=0)^(T-1) B(rho_t).
$

An environment is open-ended if some agent satisfies $overline(B)_T(pi)=Omega(T)$ @xu2026. We use the standard eventual meaning: there are $c>0$ and $T_0$ such that $overline(B)_T(pi)>=c T$ for every integer $T>=T_0$. This is an unusually demanding scaling: because $overline(B)_T$ is already an average, the sum of the per-round prices must grow quadratically. Our static theorems act on each $B(rho_t)$; only a separate policy construction can establish the required sequential reward path.

= Reward semantics

Expected reward is cardinal. A pointwise increasing map preserves deterministic payoff order, but unless it is affine it can change comparisons between lotteries. Because the frontier optimizes over randomized source-action channels, this familiar expected-utility fact becomes a structural condition on the bit-equivalent @vonneumann1944 @herstein1953.

== Positive-affine conjugacy

#statement("Theorem 1", "Positive-affine conjugacy")[
  Let $r'(theta,a)=alpha r(theta,a)+beta$ with $alpha>0$. Then
  $
    B_(cal(D)')(rho) = B_(cal(D))((rho-beta)/alpha).
  $
  Consequently, a fixed channel sequence with rewards $rho_t$ has identical bit-equivalent values after the transformed thresholds $rho'_t=alpha rho_t+beta$.
]

#proofsketch[
  For every channel $K$, the constraint $E[r']>=rho$ is equivalent to $E[r]>=(rho-beta)/alpha$. The feasible channel sets are identical and the mutual-information objective is unchanged. For a sequential policy, equality of frontier values does not by itself transport histories. If realized reward observations are also related bijectively by $R'_t=alpha R_t+beta$ and every other observation is unchanged, compose the policy with the inverse history map.
]

The restriction $alpha>0$ is essential. A negative slope reverses the reward inequality, while a zero slope erases all distinctions between thresholds. State- or action-dependent shifts are not changes of units and can also alter the frontier.

== Affine uniqueness under universal frontier conjugacy

#statement("Theorem 2", "Affine uniqueness under universal frontier conjugacy")[
  Let $J$ be an interval, let $g:J->RR$ be continuous and strictly increasing, and let $h:J->g(J)$ be one problem-independent continuous order isomorphism. Suppose that for every finite decision problem whose rewards lie in $J$,
  $
    B^(g compose r)(h(rho)) = B^r(rho)
  $
  for every $rho in J$. Then $h=g$ and $g(x)=alpha x+beta$ for some $alpha>0$.
]

#proofsketch[
  A one-state, one-action problem with constant reward $x$ has a frontier that jumps from zero to infinity at $x$. Universal conjugacy identifies $h(rho)$ with $g(rho)$. Next use a one-action problem whose source makes its reward a finite lottery $X$. Equality of the two frontier cutoffs gives $E[g(X)]=g(E[X])$ for every finite lottery. Two-point lotteries yield Jensen equality; continuity and strict increase give positive affinity. This is the frontier form of classical expected-utility uniqueness.
]

Universality is the force of the theorem. A nonlinear function may be harmless for a fixed finite problem if it agrees with an affine function on that problem's payoff support. The theorem also acts on the conditional mean reward $r(theta,a)$. Transforming a noisy realization instead generally produces $E[g(R)|theta,a] != g(E[R|theta,a])$.

This is uniqueness for exact universal frontier conjugacy, not for the coarser property of preserving only an open-ended/non-open-ended label.

== An invertible transformation reverses open-endedness

The universal theorem does not yet show that a nonlinear map changes the qualitative open-endedness classification. The following construction does.

Let $Theta_1,Theta_2,dots$ be independent fair bits. Equip their product space with its standard Borel sigma-field. The countable action set contains abstention, a query $q_i$ with deterministic reward $Theta_i$, and a prefix deployment $d_(n,x)$ for $x in {0,1}^n$ with reward

$
  r_Theta(d_(n,x)) = n 1(x=Theta_(1:n)).
$

The one-shot frontier admits every Borel source-action kernel for which the displayed expected reward is finite. After every sequential action, the realized reward is observed. A policy is a Borel randomized kernel from the observed history to the next action and must have finite per-round mean under the reward specification being evaluated. The original and cubed environments have the same actions, timing, and nonreward observations; $y mapsto y^3$ bijectively maps their reward histories.

#statement("Theorem 3", "Invertible monotone classification reversal")[
  The prefix-query environment is open-ended under $r$. Under the globally strictly increasing bijection $g(x)=x^3$, its frontier satisfies $B^(r^3)(rho)=0$ for every finite threshold $rho$. Therefore the transformed environment is non-open-ended for every policy with finite per-round transformed mean, although the two reward histories are isomorphic.
]

#proofsketch[
  Fix $0<lambda<log 2$. Under the independence reference $Q=P_Theta times P_A$, abstention has exponential moment one, a fixed query has moment $(1+e^lambda)/2<2$, and a fixed deployment has
  $
    E_Q[e^(lambda r(d_(n,x)))]
    = 1 + 2^(-n)(e^(lambda n)-1) <= 2.
  $
  The Donsker-Varadhan inequality gives $B^r(rho)>=lambda rho-log 2$ @donsker1975. A policy that alternates between querying the next bit and deploying the known prefix earns reward $k$ on deployment round $2k$. For every horizon $T$, putting $K=floor(T/2)$ and omitting the nonnegative query terms gives
  $
    overline(B)_T
    >=(lambda K(K+1)/2-K log 2)/T
    =Omega(T).
  $
  For $rho>0$, choose $n$ with $n^3>=rho$, activate the correct length-$n$ deployment with independent probability $d=rho/n^3<=1$, and abstain otherwise. Abstention and deployment have disjoint action supports, so activation is recoverable from $A$. The test channel has transformed reward $rho$ and exact information
  $
    I(Theta;A)=d n log 2 = (rho log 2)/n^2 -> 0.
  $
  Letting $n->infinity$ proves a zero frontier at every positive finite threshold; abstention handles $rho<=0$. Cubing and real cube root give inverse maps of complete reward histories, so no feedback value has been discarded.
]

The reversal is driven by two operations that ordinal language hides: expectation over lotteries and optimization over channels. The transformed test channel earns a growing payoff on a vanishing duty cycle. Feedback equivalence therefore does not imply rate-utility equivalence. More generally, the same witness works for finite-valued increasing $g$ on the nonnegative rewards with $g(0)=0$ and $g(n)/n->infinity$. Bounded nonlinear maps require problem-specific analysis and need not collapse the frontier.

= Representation semantics

A capability price should not change because the environment parameter contains irrelevant coordinates or because the same behavior has several serializations. Mutual information is representation-sensitive for an individual channel, but optimization removes this sensitivity only under exact structural conditions.

== Reward-sufficient sources

#statement("Theorem 4", "Reward-sufficient source reduction")[
  Let $S=s(Theta)$ and suppose reward factors pointwise as
  $
    r(theta,a)=overline(r)(s(theta),a).
  $
  Let $cal(D)_S$ use source $S$, the same action space, and reward $overline(r)$. Then
  $
    B_(cal(D))(rho)=B_(cal(D)_S)(rho)
  $
  for every threshold.
]

#proofsketch[
  Any channel from $S$ to $A$ lifts through $S=s(Theta)$ and satisfies $I(Theta;A)=I(S;A)$. Conversely, average an arbitrary $Theta$-channel conditional on $S$. The averaged channel preserves the joint law of $(S,A)$ and hence reward, while data processing gives $I(S;A)<=I(Theta;A)$. Taking both infima proves equality.
]

Independence of discarded coordinates is unnecessary for this static equality. If $Theta=(Z,D)$ and reward depends only on $Z$, then $B_((Z,D))=B_Z$ even when $D$ is correlated with $Z$. Correlation can still make $D$ a useful acquisition-time proxy, so the theorem does not equate learning histories.

Exact pointwise factorization is essential. Replacing reward inside a fiber by its conditional mean can destroy feasible behavior. If $S$ is constant, $U$ is a fair bit, and $r((S,U),a)=1(a=U)$, then every fixed action has conditional mean $1/2$ given $S$, but the full-source channel $A=U$ earns one.

For a finite or countable action set, the reward profile

$ S_min(theta) = (r(theta,a))_(a in cal(A)) $

is a canonical reward-sufficient statistic, augmented by state-dependent feasibility whenever feasibility is part of the decision problem. This is closely related to sufficiency, indirect rate-distortion, and decision-focused compression @dobrushin1962 @witsenhausen1980 @arumugam2021.

== Behavioral action quotients

#statement("Theorem 5", "Behavioral action quotient")[
  Let $q:cal(A)->overline(cal(A))$ satisfy
  $
    r(theta,a)=overline(r)(theta,q(a)).
  $
  Then pushing actions through $q$ gives
  $
    B_(Theta,overline(cal(A)),overline(r))(rho)
    <= B_(Theta,cal(A),r)(rho).
  $
  Equality holds if every admissible quotient action has a source-independent admissible stochastic lift supported on its $q$-fiber.
]

#proofsketch[
  Push a raw channel through $q$. Reward is unchanged and data processing cannot increase information. For the reverse direction, lift a quotient action $overline(A)$ by a kernel $Lambda(d a|overline(a))$ that is independent of $Theta$ and supported on the correct fiber. Then both $Theta->overline(A)->A$ and $overline(A)=q(A)$ hold, so the two mutual informations are equal.
]

In a finite unrestricted problem, a canonical representative supplies the lift. In a constrained problem, mere set-theoretic surjectivity is insufficient. For example, suppose a quotient has one zero-reward action, its raw fiber is ${a_0,a_1}$, and source-dependent feasibility permits only $a_theta$ when $Theta$ is a fair bit. The quotient constant action costs zero information, while every admissible raw action reveals $Theta$ and costs $log 2$. General measurable spaces require a measurable stochastic lift that also preserves dynamics, resources, and feasibility.

The result is not the same as saying that every source or action representation is Blackwell-equivalent. Blackwell order applies when one statistical experiment is a garbling of another for the fixed decision problem @blackwell1953. The unrestricted frontier already optimizes direct channels from the source; the quotient theorem instead says when different action realizations implement the same reward behavior.

= Nondegeneracy and collapse

The frontier is informative only if a genuine reward advantage carries a positive information price. The zero-information baseline $R_0$ identifies the correct comparison. Two different pathologies remain: escape in reward magnitude can collapse a fixed positive gap, while escape in the action topology can prevent boundary attainment even for bounded rewards.

== Attainment matters

Mutual information is zero exactly when $Theta$ and $A$ are independent. Hence an information-zero channel attains threshold $rho$ exactly when a source-independent action distribution attains reward at least $rho$. The equality $B(rho)=0$ need not imply that any information-zero channel is feasible: zero may be only an infimum.

#figure(
  table(
    columns: (1.2fr, 1fr, 2.2fr),
    align: (left, center, left),
    table.header([#strong[Case]], [#strong[Frontier status]], [#strong[Meaning]]),
    [Attained zero], [$B(rho)=0$], [An independent action distribution reaches $rho$],
    [Boundary nonattainment], [$rho=R_0$, $B(rho)=0$], [No independent maximizer exists; positive-information channels approach the boundary],
    [Positive-gap collapse], [$rho>R_0$, $B(rho)=0$], [Every feasible channel uses information, but a sequence drives the infimum to zero],
    [Infeasibility], [$B(rho)=+infinity$], [No channel reaches the threshold],
  ),
  caption: [Four cases hidden by the scalar frontier value. Positive-gap collapse and boundary nonattainment have different mechanisms and require different assumptions.],
) <tab:taxonomy>

== A bounded-reward certificate

Use total variation

$ norm(P-Q)_"TV" = sup_E |P(E)-Q(E)| = 1/2 norm(P-Q)_1. $

#statement("Theorem 6", "Bounded positive-gap certificate")[
  Suppose $r(theta,a) in [m,M]$ and $L=M-m>0$. Then for every $rho>R_0$,
  $
    B(rho) >= 2 ((rho-R_0)/L)^2.
  $
]

#proofsketch[
  For a channel, let $P=P_(Theta,A)$ and $Q=P_Theta times P_A$. The action is independent of the source under $Q$, so $E_Q[r]<=R_0$. A function with oscillation $L$ satisfies $E_P[r]-E_Q[r]<=L norm(P-Q)_"TV"$. Pinsker's inequality in nats gives $norm(P-Q)_"TV"<=sqrt(I(Theta;A)/2)$. Rearrangement and infimization prove the claim @pinsker1964 @csiszar1967.
]

The strict gap is indispensable. Bounded reward alone does not force attainment at $rho=R_0$. Let $Theta$ be a fair sign, let actions be $(n,s)$ for $n>=3$ and $s in {-1,+1}$, and set

$ r(theta,(n,s)) = 1 - 1/n + (theta s)/2. $

Every independent fixed action has mean $1-1/n$, so $R_0=1$ is not attained. Choosing $P(s=Theta)=1/2+1/n$ earns exactly one, while

$ I(Theta;s)=log 2-h(1/2+1/n) tilde 2/n^2 -> 0. $

Thus $B(R_0)=0$ as a nonattained infimum. The issue is topological, not a reward tail.

== Uniform integrability diagnoses positive-gap collapse

#statement("Theorem 7", "Tail escape is necessary for positive-gap collapse")[
  Fix a common measurable source-action problem, source prior $P_Theta$, reward function $r$, and source-independent baseline $R_0$. Let $P_n$ be joint laws induced by channels on that problem, let $Q_n=P_Theta times P_(A,n)$, and suppose $I_(P_n)(Theta;A)->0$. If reward is uniformly integrable under both law families,
  $
    lim_(K->infinity) sup_n E_(P_n)[|r|1(|r|>K)] = 0,
  $
  and the same display holds with $Q_n$, then
  $
    E_(P_n)[r]-E_(Q_n)[r] -> 0.
  $
  Consequently, any sequence with reward at least a fixed $rho>R_0$ and information tending to zero must violate this two-law uniform-integrability condition.
]

#proofsketch[
  Pinsker gives $norm(P_n-Q_n)_"TV"->0$. Truncate reward at $plus.minus K$. The expectation difference of the bounded part is at most $2K norm(P_n-Q_n)_"TV"$; the two tail terms vanish uniformly as $K->infinity$. Because $E_(Q_n)[r]<=R_0$, a fixed positive gap is impossible under the stated tail control. For related uniform-integrability convergence results with varying measures, see @feinberg2018.
]

Both law families are needed. Tail control only under $P_n$ does not control rewards that appear after breaking source-action dependence, and tail control only under $Q_n$ does not control rewards concentrated on increasingly rare correctly matched actions. Conversely, failure of uniform integrability is not sufficient for collapse: an unbounded source-independent reward sequence can fail uniform integrability while creating no dependent reward advantage.

== Rare bursts and increasing leverage

The basic collapse mechanism is Bernoulli time sharing. Suppose a baseline action earns $b$, and channels $K_n$ earn $R_n>b$ at information $J_n$. Activate $K_n$ with an independent probability $d$ and otherwise use the baseline. Reward is $b+d(R_n-b)$ and information is at most $d J_n$. Therefore, if $R_n->infinity$ and $J_n/(R_n-b)->0$, then $B(rho)=0$ for every finite $rho>b$.

A strict-margin example shows why poor uninformed behavior is not enough. Let $Theta$ be uniform on $[q]$. Actions are abstention and $(M,j)$; reward is $M u$ if $j=Theta$ and $-M c$ otherwise, with $c>u/(q-1)$. Every uninformed nonabstaining action has negative mean, so $R_0=0$. Output $(M,Theta)$ with probability $d_M=rho/(M u)$ and abstain otherwise. Then

$ E[r]=rho, quad I(Theta;A)=d_M log q -> 0. $

The upper tail retains expectation $rho$ and is not uniformly integrable. If $(M,j)$ is interpreted as deploying $M$ identical copies, expected deployment count is $d_M M=rho/u$: finite expected support alone does not block the witness.

Rare activation is not necessary. With fair $Theta in {-1,+1}$, actions $(n,s)$, and reward $n Theta s$, choose $P(s=Theta)=1/2+rho/(2n)$. Reward remains $rho$ while

$ I(Theta;s)=log 2-h(1/2+rho/(2n)) tilde rho^2/(2n^2). $

Here vanishing correlation is amplified by growing reward leverage.

== Compactness closes the boundary loophole

#statement("Proposition 5.1", "Compact bounded-continuous attainment")[
  Let the source be Polish with fixed prior, let the action space be compact metric, admit all Borel stochastic kernels, and let reward be bounded continuous. At every feasible threshold $rho$, the infimum defining $B(rho)$ is attained; $R_0$ is attained; and
  $
    B(rho)=0 quad "if and only if" quad rho<=R_0.
  $
]

#proofsketch[
  Joint laws with the fixed source marginal form a tight weakly closed family and hence a weakly compact family. Marginalization and the map $P_A mapsto P_Theta times P_A$ are weakly continuous, while relative entropy is jointly lower semicontinuous. The reward constraint is closed by bounded continuity. Thus the frontier infimum is attained. Continuity on the compact action space also attains $R_0$.
]

Compactness is a sufficient contract, not the weakest possible one. Any resource-restricted channel class must itself be weakly closed. On a compact action space, dropping reward continuity can again make the boundary infeasible or unattained.

= Independent composition and local price

The last question is how local reward-information costs combine. For finitely many independent components the answer is classical. The countable limit then exposes the local price of an infinitesimal reward increment as the structural quantity that matters.

== Finite infimal convolution

#statement("Theorem 8", "Independent finite composition")[
  Let $cal(D)_1,dots,cal(D)_n$ have finite source and action alphabets, independent sources, the full product action space, and additive reward
  $
    r(theta_(1:n),a_(1:n))=sum_(i=1)^n r_i(theta_i,a_i).
  $
  Then
  $
    B_"prod"(rho)
    = inf_(rho_1+dots+rho_n >= rho) sum_(i=1)^n B_i(rho_i).
  $
]

#proofsketch[
  For any global channel, set $rho_i=E[r_i(Theta_i,A_i)]$. Source independence and conditional-entropy subadditivity give
  $
    I(Theta_(1:n);A_(1:n)) >= sum_(i=1)^n I(Theta_i;A_i)
    >= sum_(i=1)^n B_i(rho_i).
  $
  Conversely, take approximately optimal component channels for a feasible allocation and form their product. Rewards and mutual informations add. This is Shannon's product-source theorem in reward coordinates @shannon1959.
]

For $n$ identical components, convexity gives equal allocation and

$ B_n(rho)=n B_1(rho/n). $

Independence, additivity, and product feasibility are substantive. If two components share the same hidden fair bit, one bit can support two correct actions, defeating the sum lower bound. Synergistic reward or coupled action constraints likewise invalidate scalar reward allocation.

== Countable iid composition

Let the component problem be finite and admit a pointwise null action $a_0$ with $r(theta,a_0)=0$. Let

$ r_* = E[max_(a in cal(A)) r(Theta,a)] > 0. $

For an iid source sequence $Theta_NN tilde mu^(times NN)$, define

$
  N(a)=sum_(i=1)^infinity 1(a_i!=a_0), quad
  cal(A)_"fs"={a in cal(A)^NN:N(a)<infinity}.
$

Thus "finite support" means finitely many nonnull coordinates almost surely, not a finitely supported channel distribution. Admit all kernels into $cal(A)_"fs"$ satisfying $E[N(A)]<infinity$, and define

$
  r_infinity(theta,a)=sum_(i=1)^infinity r(theta_i,a_i),
$

$
  B_infinity(rho)
  =inf_(K:E_K[N(A)]<infinity, E_K[r_infinity]>=rho)
    I_K(Theta_NN;A).
$

Since the component is finite and the null reward is pointwise zero, every admissible global reward is absolutely integrable.

Define the local information price

$
  kappa = lim_(x -> 0^+) frac(B_1(x), x)
  = inf_(0<x<=r_*) frac(B_1(x), x).
$

Convexity, the null channel, and time sharing with a finite-information maximizing channel give $0<=kappa<infinity$.

#statement("Theorem 9", "Countable local-price law")[
  Under the preceding iid, pointwise-null, additive, finite-support, and finite-expected-support contract, for every finite $rho>=0$,
  $
    B_infinity(rho)=kappa rho.
  $
]

#proofsketch[
  The null action gives $B_1(0)=0$. Convexity makes $frac(B_1(x),x)$ nondecreasing for $x>0$, so $B_1(x)>=kappa x$ there. For $x<=0$, the null channel gives $B_1(x)=0>=kappa x$; the inequality therefore applies to signed coordinate rewards. Let $C=max_(theta,a) abs(r(theta,a))$. For a global channel,
  $
    E[sum_i abs(r(Theta_i,A_i))]
    <=C E[N(A)]<infinity.
  $
  Hence Fubini gives $sum_i abs(rho_i)<infinity$ and $sum_i rho_i=R$, where $rho_i=E[r(Theta_i,A_i)]$. Projecting to the first $m$ coordinates and applying finite composition gives
  $
    I(Theta_NN;A) >= sum_(i=1)^m B_1(rho_i)
    >= kappa sum_(i=1)^m rho_i.
  $
  Let $m->infinity$ for the lower bound. For $rho>0$, choose independent component channels on the first $n$ coordinates whose information is within $epsilon/n$ of $B_1(rho/n)$ and whose reward is at least $rho/n$; play $a_0$ elsewhere. Then $B_infinity(rho)<=n B_1(rho/n)+epsilon$. Send $n->infinity$ and then $epsilon->0$. The null channel handles $rho=0$.
]

== A component with strictly positive local price

The local-price law is not merely a zero-versus-zero pathology. Let $Theta$ be a fair bit. The actions are a null action with reward zero and two guesses. A correct guess earns one and an incorrect guess earns $-c$, where

$
  c = log(5/2)/log(8/5).
$

Here $c>1$, so an uninformed guess has mean $(1-c)/2<0$ and the null action makes $R_0=0$. Averaging a channel with its bit-flipped copy preserves reward and cannot increase mutual information. A minimizing channel may therefore be taken to activate a guess with probability $q$ and, conditional on activation, to be correct with probability $p$. Writing $h(p)=-p log p-(1-p)log(1-p)$, its reward and information are

$
  x=q((1+c)p-c), quad
  I=q(log 2-h(p)).
$

Let $F(p)=log 2-h(p)$ and $p_*=4/5$. The derivative is $F'(p)=log(p/(1-p))$, and the choice of $c$ makes the line through the origin tangent at $p_*$:

$
  (log 2-h(p_*))/((1+c)p_*-c)
  = log 4/(1+c)
  = log(8/5).
$

Convexity gives the exact tangent inequality $F(p)>=kappa((1+c)p-c)$ for every $p$, with equality at $p_*$. Thus, if a channel feasible at threshold $x$ earns $y>=x$, symmetrization and the tangent bound give $I>=kappa y>=kappa x$. Conversely, fixing $p=p_*$ and varying $q$ traces this line. Therefore, with

$ x_*=(1+c) 4/5-c>0, $

the exact linear range is $0<=x<=x_*$ and

$ B_1(x)=kappa x, quad kappa=log(8/5)>0. $

The countable iid product consequently satisfies $B_infinity(rho)=rho log(8/5)$ for every finite $rho>=0$. The example isolates what the theorem measures: a nonzero marginal price for reward-relevant information, even though a null action permits arbitrarily thin allocation across components.

The theorem gives a dichotomy. If $kappa>0$, the countable static frontier is linear. If $kappa=0$, every finite target collapses. The number of available tasks is not enough: task count matters only through the component frontier near zero. Nor does linearity construct an open-ended learner. It is an information requirement conditional on successful behavior.

The assumptions cannot be suppressed. If all coordinates share one fair bit, activating $N$ correct copies with probability $rho/N$ holds reward and expected support fixed while information tends to zero, even when the component has a positive Donsker-Varadhan slope. A merely mean-zero null action can leave an undefined infinite reward series. Removing finite expected support without another absolute-integrability condition can also make expected reward undefined, while a hard support cap blocks the spreading argument used for achievability.

= Consequences for definitions of open-ended learning

The results suggest a semantic contract for any use of the bit-equivalent as a capability price.

1. *Fix a cardinal reward scale.* Positive-affine changes of units are universally safe. Arbitrary monotone transformations are not, even when they preserve feedback invertibly.

2. *Declare a payoff-sufficient source.* Payoff-extraneous coordinates may be quotiented without changing the optimized static frontier. Whether they belong in a broader semantic target remains a modeling choice, and they may still affect how information is acquired.

3. *Price behavioral actions.* Raw encodings should be quotiented only when quotient behavior has a measurable, feasible, resource-preserving, source-independent lift.

4. *Control tails and topology.* Boundedness, or uniform integrability under both laws along any proposed vanishing-information sequence, rules out positive-gap collapse. Compact-continuous structure, or a suitable coercive replacement, is needed to close the boundary-attainment loophole.

5. *State the composition contract.* Independent sources, additive rewards, product actions, and an integrability convention are not notation; they determine whether local information prices add.

6. *Separate frontier geometry from agent achievability.* A lower bound on successful action information does not identify an interaction protocol or a learner that can attain a growing reward path.

These conditions change how an open-endedness claim should be interpreted. The bit-equivalent is not an ordinal measure of capability. It is a rate-utility price relative to a prior, a cardinal reward, a behavioral action space, and an admissible channel class. Under a declared contract that can be a coherent and useful object. Without one, isomorphic reward histories can induce opposite classifications, or a fixed positive reward advantage can cost zero information as an infimum.

= Related work and positioning

== Rate-distortion, rate-utility, and rational inattention

Shannon's rate-distortion function minimizes mutual information subject to expected distortion and already contains finite convexity, the zero-rate cutoff, and product-source allocation @shannon1959. Replacing distortion by a constant minus reward gives the present static frontier. Genewein et al. explicitly describe the dual decision object as a rate-utility function and interpret it as the minimum processing capacity required for an expected-utility target @genewein2015. Rational inattention and information-theoretic bounded rationality study closely related mutual-information/utility tradeoffs @sims2003 @matejka2015 @ortega2011. We therefore do not claim the static frontier or its basic convex geometry as new.

Xu, Zhu, and Van Roy introduce the bit-equivalent terminology, average bit-equivalent, and the use of linear growth as a definition of open-endedness @xu2026. Our object of analysis is that new role. Classical coding theory does not require a distortion function to remain invariant under reparameterization; a semantic criterion for capability must say which transformations preserve its meaning.

== Utility transformations and reward shaping

Positive-affine uniqueness is classical in expected-utility theory @vonneumann1944 @herstein1953. Nonlinear increasing maps preserve deterministic outcome order but can change lottery order. Dynamic reward shaping has additional freedoms tied to transition structure; Ng, Harada, and Russell characterize policy-invariant shaping for Markov decision processes @ng1999. Our transformation question is narrower and more universal: which pointwise maps conjugate every reward-information frontier? The affine answer is inherited utility theory. The paper-specific contribution is the explicit interaction between a nonlinear transform and the average-bit-equivalent classification.

== Sufficiency, experiments, and indirect rate-distortion

Reward-sufficient reduction and behavioral quotienting are data-processing arguments related to Blackwell comparison @blackwell1953, indirect rate-distortion @dobrushin1962 @witsenhausen1980, and decision-focused compression @arumugam2021. Their role here is semantic: they characterize exact reductions that leave the optimized static price unchanged. The finite results are elementary; measurable selection and admissibility are the nontrivial issues in broader spaces.

== Tail escape and composition

The rare-burst mechanism resembles zero-cost symbols, on-off transmission, and flash signaling in capacity-per-unit-cost and wideband theory @verdu1990 @verdu2002. The direction of optimization differs: we minimize source information for fixed expected reward rather than maximize communication per unit input cost. We use the analogy to name a mechanism, not to claim the same theorem. Likewise, finite infimal convolution is explicitly Shannon's product-source result. The countable limit combines that tensorization with elementary convex analysis—see @rockafellar1970 for general convex background—and is presented as a delimited corollary, not a new general tensorization principle.

The conservative novelty claim is therefore a foundations synthesis for a recent definition, centered on an explicit reward-scale classification reversal and a precise account of collapse and admissibility. No priority claim is inferred from search failure.

= Limitations and open problems

First, the frontier is prior-dependent. A capability notion intended to be observer-independent may require a minimax, capacity, or prior-robust analogue. Such an analogue could change the representation and composition theorems.

Second, most results are one-shot. Stateful control, delayed effects, and information carried by action sequences call for causal or directed-information frontiers. An action quotient must then preserve transitions and future opportunity, not merely immediate reward.

Third, the broad-space statements remain assumption-sensitive. Standard-Borel source reduction requires regular conditional probabilities; action quotienting requires a measurable admissible lift; noncompact attainment requires coercivity or another tightness mechanism; countable composition requires an explicit summability convention.

Fourth, the countable theorem is iid and additive. Nonidentical components lead to an infinite allocation problem, while dependent sources can share information across tasks. Both directions may be more relevant to realistic skill growth.

Fifth, this paper does not solve statistical identification. Estimating a frontier from black-box interaction is different from defining it, and a finite sample cannot certify an infimum over unseen rare events without tail assumptions.

Finally, a static frontier does not supply a learner. A complete theory of open-ended learning needs both a semantic information-demand object and an interaction protocol under which an agent can discover behaviors that move along it.

= Conclusion

The bit-equivalent is a classical rate-utility frontier placed in a new role. That role makes its foundations consequential. Positive-affine reward changes, exact sufficient sources, and liftable behavioral quotients are safe. General monotone reward changes are not: an invertible cubic transformation can reverse open-endedness without discarding feedback. Bounded reward, or two-law uniform integrability along any proposed vanishing-information sequence, protects a fixed positive gap; noncompactness and unbounded leverage expose two distinct ways for a zero-valued infimum to escape. Independent additive components compose classically, and their countable limit is governed by the local information price at zero rather than task count alone.

The resulting lesson is not that the bit-equivalent should be abandoned. It is that the quantity is meaningful only relative to a declared semantic contract. Once reward scale, payoff-relevant source, behavioral actions, tail regularity, and admissibility are fixed, the frontier can be a coherent price of reward-relevant information. Without that contract, the classification can depend on the cardinal reward and admissibility specification even when complete reward histories are isomorphic.

#pagebreak()
#set heading(numbering: none)
= References

#text(size: 8.8pt)[#bibliography("references.bib", title: none)]

#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1.")
= Proofs and extended constructions

#include "proofs.typ"

#pagebreak()
= Computational verification

The paper's results are analytic. The repository nevertheless contains eleven deterministic regression checks that exercise the finite frontier solver and the closed-form limiting constructions. They check affine conjugacy, reward-sufficient reduction, quotient equality and failed lifting, nonidentical finite composition, the cubic and strict-margin collapse witnesses, the Pinsker certificate, a positive local-price example, bounded boundary nonattainment, and failure of countable composition under a shared source. These checks are falsification aids, not evidence replacing proofs.

The public source, proof appendix, bibliography, executable checks, and rendered PDF are archived at #link("https://github.com/Gabriel-Kahen/infinite-rulebook/pull/18")[github.com/Gabriel-Kahen/infinite-rulebook/pull/18]. From the repository root, run `uv run pytest -q tests/test_theory_foundations.py tests/test_theory_paper.py`. At this paper version, the eleven theorem checks and three artifact-integrity checks pass. The repository also records the assumption ledger and full validation report.

#figure(
  table(
    columns: (1.3fr, 2.3fr, 1fr),
    align: (left, left, center),
    table.header([#strong[Family]], [#strong[Representative check]], [#strong[Status]]),
    [Reward semantics], [Affine frontier intervals and cubic information decay], [11/11 suite passed],
    [Representation], [Duplicated source fibers, action aliases, and missing lift], [Passed],
    [Nondegeneracy], [Pinsker bound, rare burst, and boundary nonattainment], [Passed],
    [Composition], [Nonidentical infimal convolution, positive local slope, shared-source failure], [Passed],
  ),
  caption: [Executable checks included with the paper repository. Numerical agreement is not used in any proof.],
)

= Assumption-removal map

The most important hostile examples can be summarized as follows.

#table(
  columns: (1.4fr, 1.2fr, 2.2fr),
  align: (left, left, left),
  table.header([#strong[Removed assumption]], [#strong[Outcome]], [#strong[Witness]]),
  [Positive affine slope], [Conjugacy fails], [A negative slope reverses the reward constraint],
  [Exact source factorization], [Reduction fails], [A constant statistic hides a fair payoff-relevant bit],
  [Source-independent lift], [Quotient equality fails], [State-dependent feasibility forces raw actions to reveal the source],
  [Bounded/two-law UI at positive gap], [Frontier can collapse], [Strict-margin rare bursts or increasing leverage],
  [Compactness at $R_0$], [Minimizer can fail to exist], [Bounded noncompact actions approach an unattained baseline],
  [Independent sources], [Tensorization fails], [Repeated copies share one hidden bit],
  [Pointwise null/integrability], [Global reward can be undefined], [Infinitely many mean-zero inactive rewards need not sum],
  [Unlimited finite spreading], [Countable upper bound fails], [A hard support cap blocks $n->infinity$ allocation],
)

These examples define the scope of the statements rather than auxiliary caveats. A machine-readable or formal proof development could use them as regression obligations for future generalizations.
