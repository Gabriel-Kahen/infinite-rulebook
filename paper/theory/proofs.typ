#let fullproof(body) = block(
  breakable: true,
  width: 100%,
  inset: (left: 9pt, right: 4pt, top: 3pt, bottom: 5pt),
  stroke: (left: 0.7pt + rgb("A9B2B8")),
)[
  #emph[Proof.] #body
  #align(right)[∎]
]

This appendix gives complete proofs for the statements in the main text. All
logarithms are natural. Mutual information is the extended relative entropy

$ I(X;Y) = D(P_(X,Y) || P_X times P_Y) in [0,+infinity]. $

Total variation is normalized as

$ norm(P-Q)_"TV" = sup_E abs(P(E)-Q(E)) = 1/2 norm(P-Q)_1. $

Unless a sequential policy is explicitly constructed, each result concerns a
one-shot optimized frontier. A frontier identity does not, by itself, provide
a learner that can attain the corresponding behavior.

== Preliminary frontier facts

For a finite problem, the frontier is nondecreasing because its feasible sets
shrink as the reward threshold rises. It is also convex. Indeed, for channels
$K_0,K_1$ and $t in [0,1]$, let $K_t=t K_1+(1-t) K_0$. Reward is affine in the
channel. Mutual information is convex in the channel for fixed source law:
both $P_(Theta,A)=mu K$ and $P_Theta times P_A=mu times (mu K)_A$ depend
affinely on $K$, so joint convexity of relative entropy gives

$ I(K_t) <= t I(K_1) + (1-t) I(K_0). $

Mixing arbitrarily close minimizers and sending their approximation errors to
zero proves convexity of $B$ as an extended-real function.

Mutual information is zero exactly when $P_(Theta,A)=P_Theta times P_A$.
Thus an information-zero channel attains threshold $rho$ exactly when a
source-independent action distribution attains that reward. In a finite
action space, the reward of such a distribution is a convex combination of
constant-action rewards, so some constant action does at least as well. This
is an attainment statement: $B(rho)=0$ may instead be a nonattained infimum.

== Theorem 1: Positive-affine conjugacy

Assume $r'(theta,a)=alpha r(theta,a)+beta$, where $alpha>0$, the admissible
channel class is unchanged, and all displayed expectations exist.

#fullproof[
  For every channel $K$,

  $ R_(r')(K) = alpha R_r(K)+beta. $

  Positivity of $alpha$ gives

  $ R_(r')(K)>=rho quad "if and only if" quad R_r(K)>=(rho-beta)/alpha. $

  The two frontier infima therefore range over exactly the same channels, and
  their mutual-information objectives are identical. Hence

  $ B_(r')(rho)=B_r((rho-beta)/alpha). $
]

The static equality transports a sequential policy only under an additional
observation-level hypothesis. Suppose timing, actions, dynamics, nonreward
observations, and policy randomization are unchanged, and observed realized
rewards obey $Y'_t=alpha Y_t+beta$. Reward histories are then in bijection.
Compose the original decision rule with $y=(y'-beta)/alpha$ on every past
reward. Induction on time shows that the transported policy induces the same
marginal source-action channel at every round. If its two mean rewards are
$rho_t$ and $rho'_t=alpha rho_t+beta$, the static identity yields
$B_(r')(rho'_t)=B_r(rho_t)$ round by round. Clipping, coarsening, delaying, or
otherwise changing the reward observation would invalidate this transport
argument.

== Theorem 2: Affine uniqueness under universal frontier conjugacy

Let $J$ be an interval, let $g:J->RR$ be continuous and strictly increasing,
and let $h:J->g(J)$ be one problem-independent continuous order isomorphism. Assume that, for every
finite Bayesian decision problem with rewards in $J$ and every $rho in J$,

$ B^(g compose r)(h(rho))=B^r(rho). $

#fullproof[
  Fix $x in J$. A one-state, one-action problem with constant reward $x$ has
  frontier zero at $rho<=x$ and $+infinity$ at $rho>x$. Its transformed
  reward is the constant $g(x)$. Universal conjugacy therefore implies

  $ h(rho)<=g(x) quad "if and only if" quad rho<=x $

  for all $x,rho in J$. Since $h(rho) in g(J)$, write
  $h(rho)=g(y)$ for a unique $y in J$. Strict monotonicity changes the last
  equivalence into

  $ y<=x quad "if and only if" quad rho<=x $

  for every $x in J$. The two lower cuts of $J$ coincide, so $y=rho$ and
  therefore $h=g$.

  Next let a one-action finite problem have reward lottery $X$ supported in
  $J$. Put $m=E[X]$ and $m_g=E[g(X)]$. Intervals are closed under finite
  convex combinations, so $m in J$ and $m_g in g(J)$. Write $m_g=g(y)$.
  The original one-action frontier jumps at $m$, while the transformed
  frontier, evaluated at $g(rho)$, jumps at $y$. Equality of the frontiers at
  every threshold gives

  $ rho<=m quad "if and only if" quad rho<=y $

  for every $rho in J$, hence $y=m$. Thus

  $ E[g(X)] = g(E[X]) $

  for every finite lottery on $J$. Apply this identity to a two-point lottery
  taking $x,z in J$ with probabilities $t$ and $1-t$. For every
  $t in [0,1]$,

  $ g(t x+(1-t)z)=t g(x)+(1-t)g(z). $

  Hence $g$ is affine on $J$. Strict increase makes its slope positive, so
  $g(x)=alpha x+beta$ with $alpha>0$. Theorem 1 proves the converse.
]

The universal quantifier is essential. A nonlinear map can preserve a fixed
finite problem if it agrees with an affine map on that problem's payoff
support. The theorem acts pointwise on the conditional mean-payoff function.
For noisy realized reward $Y$, generally
$E[g(Y)|theta,a] != g(E[Y|theta,a])$.

== Theorem 3: Invertible monotone classification reversal

Let $Theta=(Theta_1,Theta_2,dots)$ consist of iid fair bits, with its product
standard Borel structure. The countable action set contains abstention $bot$, queries $q_i$, and deployments
$d_(n,x)$ for $x in {0,1}^n$. Reward is deterministic:

$
  r_theta(bot)=0,
  quad r_theta(q_i)=theta_i,
  quad r_theta(d_(n,x))=n 1(x=theta_(1:n)).
$

The one-shot frontier admits all Borel source-action kernels with finite mean
reward. In the sequential bandit, the realized reward is observed after each
action and every per-round mean under the active reward specification is
finite. Actions, timing, and nonreward observations are unchanged between the
two specifications.

#fullproof[
  *Original-frontier bound.* Fix $0<lambda<log 2$, a channel-induced joint
  law $P=P_(Theta,A)$, and its independence reference
  $Q=P_Theta times P_A$. For a fixed abstention action,

  $ E_Theta[e^(lambda r(Theta,bot))]=1. $

  For a fixed query,

  $ E_Theta[e^(lambda r(Theta,q_i))]=(1+e^lambda)/2<2. $

  For a fixed deployment,

  $
    E_Theta[e^(lambda r(Theta,d_(n,x)))]
    =1+2^(-n)(e^(lambda n)-1)<=2,
  $

  because $e^lambda<2$. Averaging these fixed-action moments under $P_A$
  gives $E_Q[e^(lambda r)]<=2$. The Donsker-Varadhan variational inequality
  yields

  $
    I(Theta;A)=D(P||Q)
    >= lambda E_P[r]-log E_Q[e^(lambda r)]
    >= lambda E_P[r]-log 2.
  $

  For a version of the variational formula stated only for bounded test
  functions, apply it to $min(lambda r,M)$ and let $M->infinity$. Taking the
  infimum over channels with reward at least $rho$ gives

  $ B^r(rho)>=lambda rho-log 2. $

  *Sequential witness.* At round $2k-1$, query bit $k$. Its deterministic
  reward reveals $Theta_k$. At round $2k$, deploy the known prefix
  $d_(k,Theta_(1:k))$, earning reward exactly $k$. If $rho_t$ is the mean
  reward of the induced marginal channel and

  $ overline(B)_T=1/T sum_(t=1)^T B^r(rho_t), $

  then, with $K=floor(T/2)$, nonnegativity of the omitted query-round terms
  and the preceding bound give

  $
    overline(B)_T
    >= 1/T sum_(k=1)^K (lambda k-log 2)
    = (lambda K(K+1)/2-K log 2)/T
    = Omega(T).
  $

  This establishes open-endedness under the convention in the main text.

  *Collapse after cubing.* Fix $rho>0$ and choose $n$ with $n^3>=rho$. Let
  $C$ be an independent Bernoulli variable with

  $ P(C=1)=d=rho/n^3. $

  When $C=1$, output $A=d_(n,Theta_(1:n))$; otherwise output $A=bot$. The
  transformed mean is $d n^3=rho$. Activation is recoverable from the action
  because abstention and deployment have disjoint supports. Conditional on
  activation, the action reveals exactly $n$ fair bits. Hence

  $
    I(Theta;A)
    =I(Theta;C,A)
    =I(Theta;C)+I(Theta;A|C)
    =d n log 2
    =(rho log 2)/n^2.
  $

  Letting $n->infinity$ and using nonnegativity of mutual information proves
  $B^(r^3)(rho)=0$. Abstention proves the same equality for $rho<=0$. These
  are frontier test channels, not behaviors claimed to be learnable by the
  alternating policy.

  Finally, $x mapsto x^3$ is a bijection and the feedback is deterministic.
  The original realized reward is recovered from the transformed reward by
  the real cube root, and conversely by cubing. This pointwise inverse extends
  to a bijection of complete reward histories, while the other history fields
  are identical. The frontier geometry and open-endedness classifications
  nevertheless differ.
]

The collapse uses unbounded prefix rewards and the superlinear ratio
$g(n)/n->infinity$. It does not imply that every bounded nonlinear
transformation collapses a frontier.

== Theorem 4: Reward-sufficient source reduction

Let $S=s(Theta)$ in a finite problem, admit all randomized kernels, and assume
the pointwise factorization

$ r(theta,a)=overline(r)(s(theta),a). $

#fullproof[
  Given a reduced channel $L(a|s)$, lift it by
  $K(a|theta)=L(a|s(theta))$. The reduced and lifted channels have the same
  reward. Since $Theta->S->A$ and $S$ is a function of $Theta$,

  $
    I(Theta;A)
    =I(S,Theta;A)
    =I(S;A)+I(Theta;A|S)
    =I(S;A).
  $

  Every reduced witness thus supplies a full-source witness at the same cost,
  proving $B_Theta(rho)<=B_S(rho)$.

  Conversely, given a full-source channel $K$, define its fiber average by

  $
    overline(K)(a|s)
    =sum_(theta:s(theta)=s) P(theta|S=s) K(a|theta)
  $

  for source values of positive probability, arbitrarily elsewhere. This
  reduced channel reproduces the original joint law of $(S,A)$, and therefore
  reproduces reward. Data processing gives

  $ I_(overline(K))(S;A)=I_K(S;A)<=I_K(Theta;A). $

  Thus $B_S(rho)<=B_Theta(rho)$. Combining the two inequalities proves the
  claim.
]

Exact factorization cannot be replaced by conditional-mean aggregation. Let
$S$ be constant, let $Theta=(S,U)$ with $U$ a fair bit, and set
$r(Theta,a)=1(a=U)$. Every fixed action has conditional mean $1/2$ given
$S$, but the full-source channel $A=U$ earns one. In a restricted channel
class, both lifting and fiber averaging must also preserve admissibility.

== Theorem 5: Behavioral action quotient

Let $q:cal(A)->overline(cal(A))$ preserve reward pointwise:

$ r(theta,a)=overline(r)(theta,q(a)). $

#fullproof[
  Given any raw channel, set $overline(A)=q(A)$. Reward is unchanged, and data
  processing in $Theta->A->overline(A)$ gives

  $ I(Theta;overline(A))<=I(Theta;A). $

  Taking infima proves

  $ B_(Theta,overline(cal(A)),overline(r))(rho)
     <=B_(Theta,cal(A),r)(rho). $

  For equality, assume a source-independent admissible kernel
  $Lambda(d a|overline(a))$ supported on $q^(-1)(overline(a))$ and preserving every
  stated feasibility or resource constraint. Given a quotient channel,
  sample $A$ from this lift after $overline(A)$. Then

  $ Theta->overline(A)->A quad "and" quad overline(A)=q(A) $

  almost surely. Data processing in both directions gives

  $ I(Theta;A)<=I(Theta;overline(A))<=I(Theta;A). $

  Thus reward and information are exactly preserved. Lifting arbitrarily
  close quotient minimizers proves the reverse frontier inequality and hence
  equality.
]

The lift assumption is substantive. Let $Theta$ be a fair bit, let the
quotient contain one zero-reward action $overline(a)$, and let its raw fiber be
${a_0,a_1}$, also with zero reward. Impose state-dependent feasibility that
permits only $A=a_Theta$. The quotient constant costs zero information, but
the only admissible raw channel reveals $Theta$ and costs $log 2$. No common
source-independent representative is feasible. Similarly, a quotient action
with no raw preimage can make the quotient frontier strictly smaller.

== Theorem 6: Bounded positive-gap certificate

Assume $r(theta,a) in [m,M]$, let $L=M-m>0$, and take $rho>R_0$.

#fullproof[
  For any channel, let $P=P_(Theta,A)$ and
  $Q=P_Theta times P_A$. Under $Q$ the action is independent of the source,
  so $E_Q[r]<=R_0$. If a measurable function $f$ lies in $[m,M]$, apply the
  variational characterization of total variation to $(f-m)/L in [0,1]$ to
  obtain

  $ abs(E_P[f]-E_Q[f])<=L norm(P-Q)_"TV". $

  Therefore every channel with $E_P[r]>=rho$ satisfies

  $ rho-R_0<=L norm(P-Q)_"TV". $

  Pinsker's inequality in nats gives

  $
    norm(P-Q)_"TV"
    <=sqrt(D(P||Q)/2)
    =sqrt(I(Theta;A)/2).
  $

  Rearranging and taking the infimum over feasible channels proves

  $ B(rho)>=2((rho-R_0)/L)^2. $
]

The strict gap cannot be removed. Let $Theta$ be a fair sign, let actions be
$(n,s)$ with $n>=3$ and $s in {-1,+1}$, and set

$ r(theta,(n,s))=1-1/n+(theta s)/2. $

Every source-independent action distribution has mean below one, while such
means approach one; hence $R_0=1$ and is not attained. For fixed $n$, choose
$P(s=Theta)=1/2+1/n$. Then $E[Theta s]=2/n$, so expected reward is exactly
one, while

$ I(Theta;s)=log 2-h(1/2+1/n) tilde 2/n^2 -> 0. $

Thus $B(R_0)=0$ as a nonattained infimum even though reward is bounded.

== Theorem 7: Tail escape is necessary for positive-gap collapse

Fix a common measurable source-action problem, source prior $P_Theta$, reward
function $r$, and source-independent baseline $R_0$. Let
$P_n=P_(Theta,A_n)$ be channel-induced joint laws and let
$Q_n=P_Theta times P_(A,n)$. Assume
$D(P_n||Q_n)=I(Theta;A_n)->0$ and uniform integrability under both families:

$
  lim_(K->infinity) sup_n E_(P_n)[abs(r) 1(abs(r)>K)]=0,
$

$
  lim_(K->infinity) sup_n E_(Q_n)[abs(r) 1(abs(r)>K)]=0.
$

#fullproof[
  Pinsker gives $norm(P_n-Q_n)_"TV"->0$. Let

  $ r^((K))=max(-K,min(r,K)). $

  Its oscillation is at most $2K$, so

  $
    abs(E_(P_n)[r^((K))]-E_(Q_n)[r^((K))])
    <=2K norm(P_n-Q_n)_"TV".
  $

  Moreover,

  $ abs(r-r^((K)))<=abs(r) 1(abs(r)>K). $

  The triangle inequality yields

  $
    abs(E_(P_n)[r]-E_(Q_n)[r])
    <= E_(P_n)[abs(r)1(abs(r)>K)]
      +2K norm(P_n-Q_n)_"TV"
      +E_(Q_n)[abs(r)1(abs(r)>K)].
  $

  For fixed $K$, let $n->infinity$; then let $K->infinity$. The two uniform-
  integrability assumptions prove

  $ E_(P_n)[r]-E_(Q_n)[r]->0. $

  Since $E_(Q_n)[r]<=R_0$, a sequence with
  $E_(P_n)[r]>=rho>R_0$ would keep this difference at least
  $rho-R_0>0$, a contradiction.
]

Both law families are necessary for this symmetric convergence lemma.

- *Control only under the induced laws is insufficient.* Let $Theta$ be a
  fair bit. With probability $1/n$, output $(n,Theta)$; otherwise abstain.
  Pay $2n$ exactly when the reported bit differs from $Theta$. Under $P_n$
  reward is identically zero and thus uniformly integrable, while
  $I(Theta;A_n)=(log 2)/n->0$. Under $Q_n$, mismatch occurs with probability
  $1/(2n)$, so expected reward is one and the expectation difference does not
  vanish.

- *Control only under the independence references is insufficient.* Let the
  source be infinitely many fair bits. With probability $2^(-m)$, report the
  correct length-$m$ prefix; otherwise abstain. Pay $2^m$ for a correct
  report. Under $P_m$, mean reward is one, the family is not uniformly
  integrable, and information is $2^(-m)m log 2->0$. Under $Q_m$, activation
  and an independent prefix match with probability $2^(-2m)$, so the tail
  expectation is $2^(-m)$ and the family is uniformly integrable.

Failure of uniform integrability is necessary, not sufficient. For example,
an unbounded source-independent reward family can fail uniform integrability
while $P_n=Q_n$ and the dependent reward advantage is identically zero.

The following strict-margin construction exhibits positive-gap collapse. Let
$Theta$ be uniform on $[q]$, with $q>=2$. Actions are abstention and $(M,j)$.
For $u>0$ and $c>u/(q-1)$, set

$
  r(theta,(M,j)) = cases(
    M u & "if " j=theta,
    -M c & "if " j!=theta,
  ).
$

Every fixed nonabstaining action has mean

$ M/q (u-(q-1)c)<0, $

so $R_0=0$. For $rho>0$ and $M>=rho/u$, activate $(M,Theta)$ with
$d_M=rho/(M u)$ and otherwise abstain. The mean reward is $rho$ and

$ I(Theta;A)=d_M log q=(rho log q)/(M u)->0. $

For every fixed cutoff, the induced upper tail retains expectation $rho$ for
all sufficiently large $M$. Thus uniform integrability fails exactly where
the theorem requires it to fail.

== Proposition 5.1: Compact bounded-continuous attainment

Let the source space $cal(T)$ be Polish with fixed Borel prior $mu$, the action
space $cal(A)$ be compact metric, all Borel stochastic kernels be admissible,
and $r:cal(T) times cal(A)->RR$ be bounded and continuous.

#fullproof[
  Represent kernels by joint laws with fixed source marginal:

  $ cal(C)_mu={P in cal(P)(cal(T) times cal(A)): P_Theta=mu}. $

  This family is tight. For $epsilon>0$, tightness of $mu$ supplies compact
  $K subset cal(T)$ with $mu(K)>1-epsilon$. Then
  $K times cal(A)$ is compact and has mass greater than $1-epsilon$ under
  every $P in cal(C)_mu$. The family is weakly closed because marginalization
  is weakly continuous. Prokhorov's theorem therefore makes $cal(C)_mu$
  weakly compact. Conversely, every member disintegrates into a Borel kernel
  because the spaces are standard Borel.

  If $P_n$ converges weakly to $P$, then $(P_n)_A$ converges weakly to $P_A$.
  The map $nu mapsto mu times nu$ is weakly continuous, and relative entropy
  is jointly lower semicontinuous. Hence

  $ P mapsto D(P||mu times P_A)=I_P(Theta;A) $

  is lower semicontinuous. Bounded continuity of $r$ makes
  $P mapsto integral r dif P$ continuous. Thus the feasible set

  $
    cal(C)_(mu,rho)
    ={P in cal(C)_mu: integral r dif P>=rho}
  $

  is closed and compact whenever nonempty, and the information objective
  attains its minimum there.

  Define $u(a)=integral r(theta,a) mu(d theta)$. If $a_n->a$, continuity of
  $r$ gives pointwise convergence $r(theta,a_n)->r(theta,a)$, and boundedness
  permits dominated convergence. Hence $u$ is continuous and attains its
  maximum on compact $cal(A)$. This maximum is $R_0$.

  If $rho<=R_0$, a maximizing constant action is feasible and has zero
  information. Conversely, if $B(rho)=0$, compact attainment supplies a
  minimizing joint law $P$ with $I_P(Theta;A)=0$. It factors as
  $mu times P_A$, so its reward is at most $R_0$. Therefore

  $ B(rho)=0 quad "if and only if" quad rho<=R_0. $
]

Compactness or another direct attainment condition is necessary at the
boundary. The bounded noncompact example following Theorem 6 has
$B(R_0)=0$ without an information-zero minimizer. Continuity also matters:
on the compact one-state action space
$cal(A)={0} union {1/n:n>=1}$, set $r(1/n)=1-1/n$ and $r(0)=0$. Then
$R_0=1$ is not attained and threshold one is infeasible.

== Theorem 8: Independent finite composition

For $i=1,dots,n$, let $cal(D)_i$ be finite problems. Assume independent
sources, the full product action space, all global stochastic kernels, and
additive reward

$ r(theta_(1:n),a_(1:n))=sum_(i=1)^n r_i(theta_i,a_i). $

#fullproof[
  *Converse.* Take an arbitrary global channel; its action coordinates may be
  correlated and may each depend on the entire source vector. Put

  $ rho_i=E[r_i(Theta_i,A_i)]. $

  Source independence gives
  $H(Theta_(1:n))=sum_i H(Theta_i)$. Conditional entropy is subadditive, and
  conditioning on the full action vector cannot increase entropy relative to
  conditioning only on $A_i$:

  $
    H(Theta_(1:n)|A_(1:n))
    <=sum_(i=1)^n H(Theta_i|A_(1:n))
    <=sum_(i=1)^n H(Theta_i|A_i).
  $

  Subtracting gives

  $ I(Theta_(1:n);A_(1:n))>=sum_(i=1)^n I(Theta_i;A_i). $

  The marginal law of $(Theta_i,A_i)$ defines a valid component channel even
  if the global $A_i$ depended on all source coordinates. Therefore

  $
    I(Theta_(1:n);A_(1:n))
    >=sum_(i=1)^n B_i(rho_i).
  $

  A channel feasible at total threshold $rho$ has
  $sum_i rho_i>=rho$, which proves the lower bound by taking its infimum.

  *Achievability.* Fix a feasible allocation with every $B_i(rho_i)$ finite
  and let $epsilon>0$. For each component choose a channel with reward at
  least $rho_i$ and information at most
  $B_i(rho_i)+epsilon/n$. Run these channels independently. Rewards add and
  product mutual information adds, giving total reward at least $rho$ and

  $ I(Theta_(1:n);A_(1:n))<=sum_(i=1)^n B_i(rho_i)+epsilon. $

  Let $epsilon->0$ and then infimize over allocations. If the allocation
  infimum is infinite, the converse already gives equality. Hence

  $
    B_"prod"(rho)
    =inf_(rho_1+dots+rho_n>=rho) sum_(i=1)^n B_i(rho_i).
  $
]

For identical components, convexity and monotonicity give

$
  sum_(i=1)^n B_1(rho_i)
  >=n B_1((sum_i rho_i)/n)
  >=n B_1(rho/n).
$

Equal allocation attains the allocation infimum, including under the
extended-real convention, so $B_n(rho)=n B_1(rho/n)$.

Independence is essential. If $Theta_1=Theta_2=Z$ is one fair bit and each
component pays one for a correct guess, two correct actions cost only
$I(Z;Z)=log 2$, whereas the independent-product formula at total reward two
gives $2 log 2$. Product feasibility is needed for the reverse inequality,
and nonadditive synergistic reward cannot in general be represented by scalar
reward allocations.

== Theorem 9: Countable local-price law

Let the finite component problem have a pointwise null action $a_0$ with
$r(theta,a_0)=0$ for every $theta$, and let

$ r_* = E[max_(a in cal(A)) r(Theta,a)]>0. $

Take $Theta_NN tilde mu^(times NN)$. Let

$
  cal(A)_"fs"
  ={a in cal(A)^NN: N(a)=sum_(i=1)^infinity 1(a_i!=a_0)<infinity},
$

and admit all kernels into this finite-support action space that satisfy
$E[N(A)]<infinity$. Define the ordinary additive reward

$ r_infinity(theta,a)=sum_(i=1)^infinity r(theta_i,a_i) $

and

$
  B_infinity(rho)
  =inf_(K:E_K[N(A)]<infinity, E_K[r_infinity]>=rho)
    I_K(Theta_NN;A).
$

Finally set

$
  kappa
  =lim_(x->0^+) frac(B_1(x),x)
  =inf_(0<x<=r_*) frac(B_1(x),x).
$

#fullproof[
  *Existence of the local slope.* The constant null channel gives
  $B_1(0)=0$. A pointwise maximizing action rule attains $r_*$ and has finite
  information because both alphabets are finite. Mixing it with the null
  channel makes every $x in [0,r_*]$ feasible at finite cost. For
  $0<x<y<=r_*$, convexity gives

  $ B_1(x)<=x/y B_1(y). $

  Thus $frac(B_1(x),x)$ is nondecreasing, its right limit at zero exists and equals
  the displayed infimum, and $0<=kappa<infinity$. Consequently

  $ B_1(x)>=kappa x $

  for $0<x<=r_*$. For $x<=0$, the null channel gives $B_1(x)=0$, so the same
  inequality remains valid because $kappa>=0$. No component channel can earn
  more than $r_*$.

  *Lower bound.* Let a global admissible channel be given, define
  $rho_i=E[r(Theta_i,A_i)]$, and put
  $C=max_(theta,a) abs(r(theta,a))$. Pointwise nullity and finite expected
  support imply

  $
    E[sum_(i=1)^infinity abs(r(Theta_i,A_i))]
    <=C E[N(A)]<infinity.
  $

  Tonelli and Fubini therefore justify absolute convergence and

  $ R=E[r_infinity(Theta_NN,A)]=sum_(i=1)^infinity rho_i. $

  For each finite $m$, coordinate projection, finite composition, and the
  component frontier bounds give

  $
    I(Theta_NN;A)
    >=I(Theta_(1:m);A_(1:m))
    >=sum_(i=1)^m I(Theta_i;A_i)
    >=sum_(i=1)^m B_1(rho_i)
    >=kappa sum_(i=1)^m rho_i.
  $

  Although $A_i$ may depend on the full source, its marginal joint law with
  $Theta_i$ defines a valid component channel. Letting $m->infinity$ and using
  absolute convergence gives $I(Theta_NN;A)>=kappa R$. Every channel feasible
  at $rho>=0$ has $R>=rho$, so $B_infinity(rho)>=kappa rho$.

  *Upper bound.* Fix $rho>0$, $epsilon>0$, and sufficiently large $n$ that
  $rho/n<=r_*$. On each of the first $n$ coordinates run an independent
  component channel with reward at least $rho/n$ and information at most
  $B_1(rho/n)+epsilon/n$. Play $a_0$ elsewhere. This channel has support size
  at most $n$, earns at least $rho$, and costs at most

  $ n B_1(rho/n)+epsilon. $

  Let $n->infinity$ and use the definition of $kappa$, then let
  $epsilon->0$. This gives $B_infinity(rho)<=kappa rho$. The all-null channel
  handles $rho=0$, proving

  $ B_infinity(rho)=kappa rho. $
]

#pagebreak()
=== Verification of the positive-local-price example

Let $Theta$ be fair on ${0,1}$, let $a_0$ pay zero, and let each guess pay
one when correct and $-c$ when incorrect, where

$ c=frac(log(5/2),log(8/5))>1. $

An independent guess has mean $(1-c)/2<0$, so the null action gives
$R_0=0$. Average any channel with the channel obtained by simultaneously
flipping the source and the two guess labels. Reward is preserved, and
convexity of mutual information in the channel means the average costs no
more. A symmetric channel has parameters $q,p in [0,1]$:

$
  P(A=a_0|Theta)=1-q,
  quad P(A=Theta|Theta)=q p,
  quad P(A!=Theta,A!=a_0|Theta)=q(1-p).
$

Its expected reward and information are

$
  x=q s(p), quad s(p)=(1+c)p-c,
  quad I=q F(p), quad F(p)=log 2-h(p).
$

Put $p_*=4/5$ and $kappa=log(8/5)$. Direct substitution gives

$ F'(p_*)=log 4=kappa(1+c) $

and $F(p_*)=kappa s(p_*)$. Since $F$ is convex, its supporting line at
$p_*$ proves

$ F(p)>=kappa s(p) $

for every $p$. Thus, if a channel feasible at threshold $x$ earns $y>=x$,
symmetrization gives $I>=kappa y>=kappa x$. Conversely, fix $p=p_*$ and vary
$q$. For

$ 0<=x<=x_*=s(p_*)=(1+c)4/5-c, $

choose $q=x/x_*$. The resulting channel has reward $x$ and information
$kappa x$. Therefore $B_1(x)=kappa x$ throughout this exact interval, which
also verifies that the countable product in the main text has the strictly
positive price $B_infinity(rho)=rho log(8/5)$.

This is a static iid result. Its assumptions are not cosmetic.

- *Dependence can destroy the lower bound.* Let every component source equal
  one fair bit $Z$. Each component has abstention and two guesses; a correct
  guess pays $+1$ and an incorrect guess pays $-c$, where $c>1$. For small
  enough $lambda>0$,

  $ (e^lambda+e^(-c lambda))/2<1. $

  The same variational argument as in Theorem 3 yields
  $B_1(x)>=lambda x$, so $kappa>=lambda>0$. Globally, for $N>=rho$, with
  probability $d=rho/N$, activate the correct guess on the first $N$ coordinates and
  otherwise abstain. It earns $rho$, has $E[N(A)]=d N=rho$, and costs

  $ I(Z;A)=d log 2=(rho log 2)/N->0. $

  Thus the dependent countable problem collapses despite a positive component
  local slope.

- *A mean-zero null action is insufficient.* If
  $r(Theta_i,a_0)=Theta_i$ for iid fair signs, playing $a_0$ on the allegedly
  inactive coordinates produces the divergent ordinary series
  $sum_i Theta_i$. The pointwise-zero condition prevents this ambiguity.

- *Summability and spreading are substantive.* Finite support almost surely
  without finite expected support, or a direct replacement such as
  $E[sum_i abs(r_i)]<infinity$, need not justify exchanging expectation and
  the infinite sum. A hard support cap blocks the sequence $n->infinity$ used
  in the upper bound.
