# Foundations of the bit-equivalent

## Status

This is a theory draft. It develops definitions, theorems, and remaining proof
obligations for a paper about the bit-equivalent itself. It does not rely on an
empirical study, and it makes no experimental claim.

The first pass works with finite Bayesian decision problems. Extensions to
standard Borel spaces should be stated only after the required compactness,
measurability, and integrability conditions are proved.

## 1. Decision problem and frontier

A finite Bayesian decision problem is a tuple

\[
\mathsf D=(\mathcal T,P_\Theta,\mathcal A,r),
\]

where \(\mathcal T\) and \(\mathcal A\) are finite sets,
\(\Theta\sim P_\Theta\), and
\(r:\mathcal T\times\mathcal A\to\mathbb R\) is mean reward. A behavioral
channel \(K(a\mid\theta)\) induces

\[
R_{\mathsf D}(K)=\mathbb E[r(\Theta,A)],
\qquad
I_{\mathsf D}(K)=I(\Theta;A).
\]

The reward--information frontier is

\[
B_{\mathsf D}(\rho)
=
\inf_{K:R_{\mathsf D}(K)\geq\rho} I_{\mathsf D}(K),
\]

with the infimum of the empty set equal to \(+\infty\). Define the best
zero-information reward

\[
R_0(\mathsf D)
=
\sup_{\nu\in\mathcal P(\mathcal A)}
\mathbb E_{P_\Theta\otimes\nu}[r(\Theta,A)]
=
\sup_{a\in\mathcal A}\mathbb E[r(\Theta,a)].
\]

The last equality holds whenever the displayed expectations are well defined.
The supremum is a maximum for finite problems, but need not be attained on a
noncompact action space. All information quantities use natural logarithms.
Our total-variation convention is

\[
\lVert P-Q\rVert_{\mathrm{TV}}
=\sup_E|P(E)-Q(E)|
=\tfrac12\lVert P-Q\rVert_1.
\]

For finite problems, \(B_{\mathsf D}\) is nondecreasing and convex on its
feasible reward interval. Monotonicity follows from nested feasible sets.
Convexity follows by mixing two channels and using convexity of mutual
information in the channel for fixed \(P_\Theta\).

## 2. Reward transformations

### Theorem 1: positive-affine invariance

Let \(\alpha>0\), \(\beta\in\mathbb R\), and

\[
r'(\theta,a)=\alpha r(\theta,a)+\beta.
\]

Then

\[
B_{\mathsf D'}(\rho)
=
B_{\mathsf D}\!\left(\frac{\rho-\beta}{\alpha}\right).
\]

In particular, any fixed sequence of behavioral channels with mean rewards
\(\rho_t\) has transformed means \(\rho'_t=\alpha\rho_t+\beta\), and

\[
B_{\mathsf D'}(\rho'_t)=B_{\mathsf D}(\rho_t)
\]

at every time.

#### Proof

For every channel \(K\),

\[
R_{\mathsf D'}(K)\geq\rho
\quad\Longleftrightarrow\quad
R_{\mathsf D}(K)\geq(\rho-\beta)/\alpha.
\]

The feasible channel sets are identical after remapping the threshold, and the
mutual-information objective is unchanged.

#### Sequential corollary

The static theorem does not by itself assert that an adaptive policy can be
run unchanged after transforming its observations. Suppose a sequential
environment either leaves observations unchanged or replaces every observed
realized reward by \(R'_t=\alpha R_t+\beta\). Reward histories are then in
bijection, so a policy can be transported by applying the inverse affine map
to its observed rewards. The transported policy induces the same action
channel at each time. Its average bit-equivalent, and hence its open-endedness
classification, is exactly unchanged.

### Theorem 2: maximal universal invariance class

Let \(J\) be an interval, let \(g:J\to\mathbb R\) be continuous and strictly
increasing, and let \(h:J\to g(J)\) be a continuous order isomorphism. Suppose,
for every finite
Bayesian decision problem with rewards in \(J\),

\[
B^{g\circ r}_{\mathsf D}(h(\rho))=B^r_{\mathsf D}(\rho)
\]

for every \(\rho\in J\). Then

\[
g(x)=\alpha x+\beta,
\qquad \alpha>0,
\]

and \(h=g\). Thus positive-affine maps are the maximal class of pointwise
transformations that preserves every reward--information frontier up to a
universal relabeling of reward.

#### Proof

A one-state, one-action problem with constant reward \(x\) has frontier zero
at \(\rho\leq x\) and infinity at \(\rho>x\). Universal conjugacy gives

\[
h(\rho)\leq g(x)
\quad\Longleftrightarrow\quad
\rho\leq x
\]

for every \(x,\rho\in J\). Because \(h(\rho)\in g(J)\), write
\(h(\rho)=g(y)\). Strict monotonicity forces \(y=\rho\), and hence \(h=g\).

Next, let a one-action problem have a finitely supported random reward
\(X=r_\Theta(a)\). Its frontier jumps from zero to infinity at
\(m=\mathbb E[X]\), while the transformed frontier jumps at
\(m_g=\mathbb E[g(X)]\). Because \(g(J)\) is an interval, write
\(m_g=g(y)\). The same cutoff argument forces \(y=m\), so

\[
\mathbb E[g(X)]=g(\mathbb E[X])
\]

for every finite lottery. Two-point lotteries give Jensen equality, so \(g\)
is affine; strict increase makes its slope positive. Theorem 1 proves the
converse.

This is a universal statement. A nonlinear transformation can still preserve
one particular problem when it agrees with an affine function on that
problem's payoff support.

The theorem transforms the conditional mean reward \(r(\theta,a)\)
pointwise. For noisy realized reward \(R\), transforming observations instead
generally produces
\(\mathbb E[g(R)\mid\theta,a]\ne g(\mathbb E[R\mid\theta,a])\); those are
different decision problems.

### Theorem 2.1: a monotone transformation can flip open-endedness

This construction is a separately proved standard-Borel extension of the
finite framework above: its source is a countable product of bits and its
action space is countable.

Let \(\Theta_1,\Theta_2,\ldots\) be independent fair bits. The action set
contains abstention, a query \(q_i\) with noiseless reward \(\Theta_i\), and a
finite prefix deployment \(d_{n,x}\), where \(x\in\{0,1\}^n\), with reward

\[
r_\Theta(d_{n,x})
=
n\,\mathbf 1\{x=\Theta_{1:n}\}.
\]

This defines a deterministic bandit process: after choosing an action, the
agent observes its realized reward. A policy is a sequence of randomized
history-dependent action kernels. At round \(t\), its marginal action channel
induces a finite mean reward \(\rho_t\), and its average bit-equivalent through
time \(T\) is \(T^{-1}\sum_{t=1}^T B^r(\rho_t)\).

For any \(0<\lambda<\log2\), the Donsker--Varadhan variational inequality
gives

\[
B^r(\rho)\geq\lambda\rho-\log2.
\]

Indeed, under the product reference law \(Q=P_\Theta P_A\), abstention has
exponential moment one, while

\[
\mathbb E[e^{\lambda r(q_i)}]=\frac{1+e^\lambda}{2}<2
\]

and

\[
\mathbb E[e^{\lambda r(d_{n,x})}]
=1+2^{-n}(e^{\lambda n}-1)\leq2
\]

for \(0<\lambda<\log2\). Averaging over \(P_A\) preserves this bound.
Donsker--Varadhan therefore yields
\(I(\Theta;A)\geq\lambda\mathbb E[r]-\log2\). An agent that alternates
between querying the next bit and deploying the known prefix earns reward
\(k\) on deployment round \(2k\). For \(T=2K\), ignoring the nonnegative
query-round terms gives

\[
\frac1{2K}\sum_{t=1}^{2K}B^r(\rho_t)
\geq
\frac1{2K}\sum_{k=1}^K(\lambda k-\log2)
=\frac{\lambda(K+1)}4-\frac{\log2}{2}
=\Omega(T).
\]

The original environment is therefore open-ended.

Now apply the globally strictly increasing bijection \(g(x)=x^3\). For any
finite \(\rho>0\), take a sequence \(n\to\infty\) with \(n^3\geq\rho\), reveal
the correct prefix in the action with probability \(d=\rho/n^3\), and abstain
otherwise. Then

\[
\mathbb E[g(r)]=\rho,
\qquad
I(\Theta;A)=d\,n\log2=\frac{\rho\log2}{n^2}\longrightarrow0.
\]

The information identity follows by conditioning on the independent
activation coin: abstention and deployment have disjoint action supports, and
the activated action reveals exactly the first \(n\) fair bits.

Abstention handles \(\rho\leq0\). Hence \(B^{r^3}(\rho)=0\) for every finite
threshold, so every policy whose
per-round transformed mean rewards are finite has zero average bit-equivalent.
Cubing is invertible and leaves the noiseless feedback information
recoverable; the classification flip comes from the geometry of expected
reward, not lost learnability.

More generally, the same witness collapses the transformed frontier for any
finite-valued strictly increasing \(g\) on the nonnegative reward range with
\(g(0)=0\) and \(g(n)/n\to\infty\). A bounded nonlinear transformation need
not collapse a frontier: transformation safety is an environment-specific
question outside the universal affine class.

## 3. Representation invariance

### Theorem 3: reward-sufficient source reduction

Let \(S=s(\Theta)\) be a deterministic statistic and suppose

\[
r(\theta,a)=\bar r(s(\theta),a).
\]

Let \(\mathsf D_S\) be the decision problem with source \(S\), the same action
set, and reward \(\bar r\). Then

\[
B_{\mathsf D}(\rho)=B_{\mathsf D_S}(\rho)
\]

for every threshold.

#### Proof

Any channel from \(S\) to \(A\) lifts to a channel from \(\Theta\) through
\(S\), preserving reward and satisfying \(I(\Theta;A)=I(S;A)\). Conversely,
given any channel from \(\Theta\) to \(A\), average it conditional on \(S\).
The averaged channel preserves the joint law of \((S,A)\), and therefore the
reward, while

\[
I(S;A)\leq I(\Theta;A)
\]

by data processing. Taking the two infima proves equality.

This theorem immediately implies invariance to augmenting a reward-sufficient
latent with arbitrary payoff-extraneous variables. Independence of the added
variables is unnecessary: if the represented source is \((Z,D)\) and reward
depends only on \(Z\), then

\[
B_{(Z,D)}(\rho)=B_Z(\rho).
\]

When \(D\) is correlated with \(Z\), however, it can be an acquisition-time
proxy for the payoff-relevant state. Thus the equality is about the optimized
static frontier; independence is still required if \(D\) is intended to be a
clean experimental distractor that cannot help learning.

For finite or countable action spaces, the reward profile

\[
S_{\min}(\theta)=\bigl(r(\theta,a)\bigr)_{a\in\mathcal A}
\]

is a canonical reward-sufficient statistic. State-dependent feasibility must
be included in this profile when it is part of the decision problem.

Exact factorization is essential. Defining
\(\bar r(s,a)=\mathbb E[r(\Theta,a)\mid S=s]\) is not sufficient: a
\(\Theta\)-dependent channel can select actions using payoff variation inside
an \(S\)-fiber. For example, let \(S\) be constant, let \(U\) be a fair bit,
let \(\Theta=(S,U)\), and set \(r(\Theta,a)=\mathbf1\{a=U\}\). Every fixed
action has conditional mean \(1/2\) given \(S\), but the channel \(A=U\)
earns one. Randomized channels are also essential because averaging a channel
within a fiber must remain admissible.

### Theorem 4: behavioral action quotient

Let \(q:\mathcal A\to\bar{\mathcal A}\) be a map such that

\[
r(\theta,a)=\bar r(\theta,q(a)).
\]

Without any lifting assumption,

\[
B_{(\Theta,\bar{\mathcal A},\bar r)}(\rho)
\leq
B_{(\Theta,\mathcal A,r)}(\rho),
\]

because pushing a raw action through \(q\) preserves reward and cannot
increase information. If \(q\) is surjective and has a source-independent
lift supported on each fiber, then

\[
B_{(\Theta,\mathcal A,r)}(\rho)
=
B_{(\Theta,\bar{\mathcal A},\bar r)}(\rho).
\]

#### Proof

For the reverse direction, lift a quotient channel by sampling a raw action
from a kernel \(\Lambda(da\mid\bar a)\) supported on
\(q^{-1}(\bar a)\). We then have both \(\Theta\to\bar A\to A\) and
\(\bar A=q(A)\), so data processing in both directions gives
\(I(\Theta;A)=I(\Theta;\bar A)\). A canonical representative supplies such a
lift in finite spaces.

For general spaces, the corresponding result requires a measurable section or
a source-independent stochastic lift that also preserves resource and
admissibility constraints. Equality can fail when purportedly equivalent
actions differ in reward, future dynamics, feasibility, or resource cost.

### Proposition 4.1: experiment-restricted Blackwell monotonicity

Suppose a hidden payoff state \(W\) is observed through an experiment \(E_i\)
before choosing an action. Let \(B_{E_i}(\rho)\) be the infimum of \(I(W;A)\)
over randomized decision kernels based on that experiment. The unrestricted
frontier above is the identity-experiment case. If \(E_1\) Blackwell-dominates
\(E_2\), so \(E_2\) is a garbling of \(E_1\), then

\[
B_{E_1}(\rho)\leq B_{E_2}(\rho).
\]

An \(E_1\)-decision maker can simulate the garbling and then use any
\(E_2\)-rule, reproducing the same joint law of \((W,A)\). Blackwell-equivalent
experiments therefore have identical frontiers. Reward-sufficient source
reduction is task-specific and sharper for this reward family: it can give
frontier equality even
when the full source and its statistic are not Blackwell-equivalent for every
possible loss function.

## 4. Nondegeneracy and collapse

### Lemma 5.0: attained zero information

A feasible channel attains \(I(\Theta;A)=0\) at threshold \(\rho\) if and
only if a source-independent action distribution attains reward at least
\(\rho\). For finite action spaces, this is equivalent to the existence of a
constant action \(a\) with

\[
\mathbb E[r(\Theta,a)]\geq\rho.
\]

Indeed, mutual information is zero exactly when the joint law factors as
\(P_\Theta P_A\), and the reward of a source-independent mixture cannot exceed
its best component. For general action spaces, every threshold
\(\rho<R_0(\mathsf D)\) has a zero-information witness, while the boundary
\(\rho=R_0(\mathsf D)\) has one if and only if the supremum is attained by an
independent action distribution. Consequently, if \(\rho>R_0(\mathsf D)\) but
\(B_{\mathsf D}(\rho)=0\), zero is a nonattained infimum: every feasible
channel has positive information, although some feasible sequence has
information tending to zero.

### Theorem 5: bounded-reward noncollapse

Suppose \(r(\theta,a)\in[m,M]\) and let \(0<L=M-m<\infty\). Then, for every
\(\rho>R_0(\mathsf D)\),

\[
B_{\mathsf D}(\rho)
\geq
2\left(\frac{\rho-R_0(\mathsf D)}{L}\right)^2.
\]

#### Proof

For a channel \(K\), let \(P=P_{\Theta,A}\) and
\(Q=P_\Theta\otimes P_A\). The action under \(Q\) is independent of the
source, so \(\mathbb E_Q[r]\leq R_0(\mathsf D)\). The variational definition
of total variation and Pinsker's inequality give

\[
\mathbb E_P[r]-R_0(\mathsf D)
\leq
L\,\lVert P-Q\rVert_{\mathrm{TV}}
\leq
L\sqrt{\frac{I(\Theta;A)}{2}}.
\]

Rearranging and taking the infimum gives the claim.

If \(L=0\), reward is constant: the frontier is zero at feasible thresholds
and \(+\infty\) above them.

For finite problems, the maximum defining \(R_0\) is attained. Hence
\(B_{\mathsf D}(\rho)=0\) exactly on the zero-information reward region
\(\rho\leq R_0(\mathsf D)\), and the frontier is strictly positive above it.

### Theorem 6: collapse requires escaping integrability

Let \(K_n\) be channels with \(I_{K_n}(\Theta;A)\to0\), and let
\(Q_n=P_\Theta\otimes P_{A,n}\). Pinsker implies
\(\lVert P_n-Q_n\rVert_{\mathrm{TV}}\to0\). Assume uniform integrability over
both varying-law families, explicitly

\[
\lim_{K\to\infty}\sup_n
\mathbb E_{P_n}[|r|\mathbf 1\{|r|>K\}]
=
\lim_{K\to\infty}\sup_n
\mathbb E_{Q_n}[|r|\mathbf 1\{|r|>K\}]
=0.
\]

Then

\[
\mathbb E_{P_n}[r]-\mathbb E_{Q_n}[r]\to0.
\]

Consequently, a sequence with reward bounded below by
\(\rho>R_0(\mathsf D)\) and information tending to zero must violate uniform
integrability. Nonattained collapse above the zero-information baseline is
therefore necessarily accompanied by uncontrolled reward tails under the
dependent laws, their independence references, or both.

To prove the convergence claim, truncate reward at \(\pm K\). The bounded
part has expectation difference at most
\(2K\lVert P_n-Q_n\rVert_{\mathrm{TV}}\); the two tail terms vanish uniformly
as \(K\to\infty\). A uniform \(L^{1+\delta}\) bound or a de la
Vallée--Poussin envelope is sufficient for the required uniform
integrability when it holds uniformly over the union
\(\{P_n,Q_n:n\geq1\}\).

### Lemma 6.1: rare-burst collapse

Let a constant baseline action have mean reward \(b\). Suppose channels
\(K_n\) attain rewards \(R_n>b\) at information costs \(J_n\). Activate
\(K_n\) with an independent Bernoulli probability \(d\), and otherwise play a
baseline action. The mixed channel has

\[
R'=b+d(R_n-b),
\qquad
I'\leq dJ_n.
\]

This follows by adjoining the independent activation coin and applying data
processing. Equality holds when activation is recoverable from the final
action, for example when the baseline and burst supports are disjoint.

Therefore, if \(R_n\to\infty\) and

\[
\frac{J_n}{R_n-b}\longrightarrow0,
\]

then \(B_{\mathsf D}(\rho)=0\) for every finite \(\rho>b\). Choose
\(d_n=(\rho-b)/(R_n-b)\). This is the general mechanism behind increasingly
valuable behavior used with increasingly small probability.

#### Strict-margin example

Let \(\Theta\) be uniform on \([q]\). The actions are abstention and pairs
\((M,j)\). Reward is \(Mu\) when \(j=\Theta\) and \(-Mc\) otherwise, where
\(c>u/(q-1)\). Every source-independent nonabstaining action has negative mean,
so \(R_0=0\). Output \((M,\Theta)\) with probability
\(d_M=\rho/(Mu)\), and otherwise abstain, taking \(M\geq\rho/u\). Then

\[
R=\rho,
\qquad
I(\Theta;A)=d_M\log q\longrightarrow0.
\]

Thus strict negative uninformed reward does not by itself prevent collapse
when informed payoff magnitude is unbounded. The reward family is not
uniformly integrable: for every fixed cutoff \(K\), its upper tail retains
expectation \(\rho\) once \(Mu>K\).

If \((M,j)\) is interpreted as deploying \(M\) identical copies of rule
\(j\), then the activated witness has expected deployment count
\(d_M M=\rho/u\). Thus finite expected support alone does not block this
collapse mechanism. A hard support cap or a uniform superlinear support
moment can do so.

Collapse need not use rare activation. With a fair sign
\(\Theta\in\{-1,+1\}\), actions \((n,s)\), and reward
\(r(\theta,(n,s))=n\theta s\), choose
\(P(s=\theta)=1/2+\rho/(2n)\) for \(n\geq\rho\). Reward stays at \(\rho\), while

\[
I(\Theta;s)
=
\log2-h\!\left(\frac12+\frac{\rho}{2n}\right)
\sim
\frac{\rho^2}{2n^2}.
\]

Here \(h\) is binary entropy in nats. Vanishing correlation is amplified by
increasing reward leverage.

### Theorem 6.2: compact bounded-reward attainment

Let the source be Polish with fixed prior, let the action space be compact
metric, admit all Borel stochastic kernels, and let reward be bounded
continuous. Then every feasible frontier value has a minimizing channel and
\(R_0\) is attained. Hence

\[
B_{\mathsf D}(\rho)=0
\quad\Longleftrightarrow\quad
\rho\leq R_0(\mathsf D).
\]

#### Proof

The joint laws with the fixed source marginal are tight and weakly closed,
hence weakly compact. Marginalization and
\(P_A\mapsto P_\Theta\otimes P_A\) are weakly continuous, while relative
entropy is jointly lower semicontinuous. Thus
\(I(\Theta;A)=D(P_{\Theta,A}\Vert P_\Theta P_A)\) is lower semicontinuous.
Bounded continuity of reward makes the feasible reward constraint closed, so
the infimum is attained. The function
\(a\mapsto\mathbb E[r(\Theta,a)]\) is continuous on a compact space, so
\(R_0\) is attained as well. A zero-valued minimum is an independent joint law
and can attain no more than \(R_0\); the maximizing constant action proves the
converse. Any additional resource or support restriction must itself define a
weakly closed subset for this argument to survive.

Bounded reward alone does not ensure attainment at the boundary on a
noncompact action space. Let \(\Theta\) be a fair sign and let actions be
\((n,s)\) for \(n\geq3\) and \(s\in\{-1,+1\}\), with

\[
r(\theta,(n,s))=1-\frac1n+\frac12\theta s.
\]

Every source-independent fixed action has mean \(1-1/n\), so \(R_0=1\) is
not attained. Fix \(n\) and choose
\(P(s=\Theta)=1/2+1/n\). The expected reward is exactly one, whereas

\[
I(\Theta;s)=\log2-h(1/2+1/n)\sim2/n^2.
\]

Thus \(B_{\mathsf D}(1)=0\) is a nonattained infimum even though reward is
globally bounded. This is a boundary phenomenon and does not contradict
Theorem 5's strict-above-baseline lower bound.

## 5. Composition

### Theorem 7: independent-product composition

Let \(\mathsf D_1,\ldots,\mathsf D_n\) have independent sources, product
actions, and additive reward

\[
r(\theta_{1:n},a_{1:n})=\sum_{i=1}^n r_i(\theta_i,a_i).
\]

Then

\[
B_{\otimes_i\mathsf D_i}(\rho)
=
\inf_{\rho_1+\cdots+\rho_n\geq\rho}
\sum_{i=1}^n B_{\mathsf D_i}(\rho_i).
\]

#### Proof

For any joint channel, set
\(\rho_i=\mathbb E[r_i(\Theta_i,A_i)]\). Source independence and the entropy
chain rule give

\[
I(\Theta_{1:n};A_{1:n})
\geq
\sum_{i=1}^n I(\Theta_i;A_i)
\geq
\sum_{i=1}^n B_{\mathsf D_i}(\rho_i).
\]

Explicitly,

\[
H(\Theta_{1:n}\mid A_{1:n})
\leq\sum_{i=1}^n H(\Theta_i\mid A_{1:n})
\leq\sum_{i=1}^n H(\Theta_i\mid A_i),
\]

while independence gives
\(H(\Theta_{1:n})=\sum_iH(\Theta_i)\). Subtracting proves the first
information inequality. A relative-entropy argument is needed beyond finite
source alphabets.

This proves the converse. For achievability, choose approximately optimal
component channels at any feasible allocation \((\rho_i)\) and take their
product. Rewards and mutual informations add.

### Corollary 7.1: identical tensorization

For \(n\) identical components,

\[
B_n(\rho)=nB_1(\rho/n).
\]

The composition theorem reduces the problem to allocating reward across
components. Convexity of \(B_1\) makes the equal allocation optimal.

### Theorem 7.2: countable tensorization and the local information price

Let \(\mathsf D=(\mathcal T,P_\Theta,\mathcal A,r)\) be finite. Assume there is
a null action \(a_0\) with \(r(\theta,a_0)=0\) for every \(\theta\), and let

\[
r_*=\mathbb E\left[\max_{a\in\mathcal A}r(\Theta,a)\right]>0.
\]

Let \(\Theta_{\mathbb N}\sim P_\Theta^{\otimes\mathbb N}\) and define

\[
\mathcal A_{\mathrm{fs}}
=
\left\{a\in\mathcal A^{\mathbb N}:
N(a):=\#\{i:a_i\ne a_0\}<\infty\right\}.
\]

Because \(\mathcal A\) is finite, \(\mathcal A_{\mathrm{fs}}\) is countable
and hence standard Borel. An admissible global channel is a stochastic kernel
from \(\mathcal T^{\mathbb N}\) to \(\mathcal A_{\mathrm{fs}}\) satisfying
finite expected support size

\[
\mathbb E N(A)<\infty,
\qquad
N(A)=\#\{i:A_i\ne a_0\}.
\]

Define

\[
r_\infty(\theta,a)=\sum_{i=1}^\infty r(\theta_i,a_i)
\]

and

\[
B_\infty(\rho)
=
\inf_{K:\,\mathbb E_KN(A)<\infty,\;
            \mathbb E_Kr_\infty(\Theta,A)\geq\rho}
I_K(\Theta_{\mathbb N};A).
\]

Finally, let

\[
\kappa
=
\lim_{x\downarrow0}\frac{B_1(x)}{x}
=
\inf_{0<x\leq r_*}\frac{B_1(x)}x.
\]

Then, for every finite \(\rho\geq0\),

\[
B_\infty(\rho)=\kappa\rho
\]

#### Proof

The null channel gives \(B_1(0)=0\). Mixing it with a finite-information
oracle channel attaining \(r_*\) makes every \(x\in[0,r_*]\) feasible.
Convexity implies that \(B_1(x)/x\) is nondecreasing for \(x>0\), so the
displayed finite limit exists and \(B_1(x)\geq\kappa x\). The same inequality
also holds for \(x<0\), because \(B_1(x)=0\geq\kappa x\).

For any admissible global channel, put
\(\rho_i=\mathbb E[r(\Theta_i,A_i)]\). Since component reward is bounded and
the null action has pointwise zero reward,

\[
\mathbb E\sum_i|r(\Theta_i,A_i)|
\leq \lVert r\rVert_\infty\mathbb E N(A)<\infty.
\]

Fubini therefore gives \(R=\sum_i\rho_i\) with absolute convergence. For
each \(m\), finite-coordinate data processing followed by Theorem 7's entropy
inequality gives

\[
I(\Theta_{\mathbb N};A_{\mathbb N})
\geq I(\Theta_{1:m};A_{1:m})
\geq\sum_{i=1}^m I(\Theta_i;A_i)
\geq\sum_{i=1}^m B_1(\rho_i)
\geq\kappa\sum_{i=1}^m\rho_i.
\]

The third term is valid even when \(A_i\) depends globally: its marginal joint
law admits a regular conditional distribution of \(A_i\) given \(\Theta_i\),
which is a valid component channel at reward \(\rho_i\).

Letting \(m\to\infty\) proves \(I\geq\kappa R\), hence the lower bound.

For the reverse inequality, fix \(\rho>0\), \(\varepsilon>0\), and any
sufficiently large \(n\) such that \(\rho/n\leq r_*\). On each of the first
\(n\) coordinates, run an independent component channel with reward at least
\(\rho/n\) and information at most
\(B_1(\rho/n)+\varepsilon/n\); play \(a_0\) elsewhere. This channel is
admissible and gives

\[
B_\infty(\rho)\leq nB_1(\rho/n)+\varepsilon.
\]

Letting \(\varepsilon\downarrow0\) and then \(n\to\infty\) gives
\(B_\infty(\rho)\leq\kappa\rho\). The all-null channel handles \(\rho=0\).

This theorem exposes the local price of reward, rather than task count, as the
structural quantity controlling this iid additive finite-expected-support
frontier. A positive \(\kappa\) yields a linear frontier; \(\kappa=0\) yields
collapse.

## 6. Open proof and novelty obligations

1. Derive the weakest frontier-level axiom under which positive-affine reward
   transformations are the maximal universal invariance class.
2. Determine which additional nonlinear transformations preserve particular
   restricted reward families, without weakening the universal theorem.
3. Extend source reduction and action quotienting to standard Borel spaces
   with explicit regular conditional distributions and measurable lifts.
4. Extend the uniform-integrability result from a necessary condition to a
   useful sufficient taxonomy under explicit tail classes.
5. Explore countable composition beyond the finite-expected-support contract.
6. Compare every statement against classical rate--distortion, value-equivalent
   control, Bayesian sufficiency, and Blackwell comparison results before
   claiming novelty.
7. Keep decision-frontier results separate from agent achievability. A linear
   frontier alone does not prove that an interaction protocol supplies an
   agent capable of attaining a linearly growing reward path.
