# Phase 0 theory contract

This note records the assumptions that the implementation treats as part of the
mathematical API. Full proofs will be expanded for the paper; the statements
below drive regression tests.

## Independent Rulebook

Let \(\Theta\) be uniform on \(\{1,\ldots,q\}\). A one-coordinate
action either abstains or predicts one label. Correct prediction earns \(u>0\),
incorrect prediction earns \(-c\), and abstention earns zero.

The strict-margin condition is

\[
c>\frac{u}{q-1}.
\]

It is required for every primary experiment.

## Proposition 1: symmetric channel

An optimal one-coordinate channel can be parameterized by deployment
probability \(d\) and conditional correctness \(p\). Simultaneously permuting
source and action labels preserves reward, and averaging over those
permutations cannot increase mutual information because mutual information is
convex in the channel for a fixed source distribution.

Define

\[
g(p)=(u+c)p-c
\]

and

\[
J_q(p)
=
p\log(qp)
+
(1-p)\log\left(\frac{q(1-p)}{q-1}\right).
\]

Then expected reward and information are

\[
R(d,p)=d g(p),
\qquad
I(d,p)=d J_q(p).
\]

## Proposition 2: one-coordinate frontier

For \(0<r\le u\),

\[
B_1(r)
=
\min_{p\ge(r+c)/(u+c)}
r\frac{J_q(p)}{g(p)}.
\]

Let \(p^\star\) minimize \(J_q(p)/g(p)\) on the profitable interval,
and define

\[
\kappa=\frac{J_q(p^\star)}{g(p^\star)},
\qquad
r^\star=g(p^\star).
\]

The frontier is

\[
B_1(r)
=
\begin{cases}
\kappa r,&0\le r\le r^\star,\\
J_q((r+c)/(u+c)),&r^\star\le r\le u,\\
+\infty,&r>u.
\end{cases}
\]

For the baseline \(q=4,u=c=1\),

\[
p^\star=\frac34,
\qquad
r^\star=\frac12,
\qquad
\kappa=\log3.
\]

## Proposition 3: tensorization

For \(N\) independent coordinates, additive reward, and an unrestricted joint
finite action,

\[
B_N(\rho)
=
N B_1(\rho/N),
\qquad
0\le\rho\le Nu.
\]

The converse follows from source independence and the entropy chain rule:

\[
I(\Theta_{1:N};A_{1:N})
\ge
\sum_i I(\Theta_i;A_i).
\]

Product channels attain equality.

## Proposition 4: infinite frontier

For countably many independent coordinates and finite-support actions with
finite expected support,

\[
B_\infty(\rho)=\kappa\rho
\]

for every finite \(\rho\ge0\). Achievability splits reward across enough
coordinates to remain in the linear segment of \(B_1\). The finite-coordinate
converse supplies the lower bound.

## Proposition 5: zero-margin collapse

If \(c<u/(q-1)\), uninformed deployment has positive reward and every finite
reward threshold is attainable at zero information.

If \(c=u/(q-1)\), predictions with advantage \(\delta\) over chance have reward
\(\Theta(\delta)\) but information \(\Theta(\delta^2)\). Distributing a fixed
reward across \(N\) coordinates makes total information \(O(1/N)\). Thus in
both cases,

\[
B_\infty(\rho)=0
\]

for every finite positive \(\rho\).

## Proposition 6: unrestricted redundancy collapse

Let all surface rules be deterministic public functions of a finite latent
core \(Z\). If arbitrarily many derived rules receive additive reward, a
stochastic channel can reveal \(Z\) with probability \(d\), deploy \(M\)
correct rules when revealed, and abstain otherwise. Choosing
\(d=\rho/(Mu)\) gives

\[
I(Z;A)\le dH(Z)\to0.
\]

Therefore the unrestricted redundant frontier is zero at every finite feasible
reward threshold. Resource-controlled redundancy is a different environment
and must declare its support or reward cap.

## Proposition 7: symbolic distractor invariance

ALEA observations are independent runtime draws and are not persistent
coordinates of \(\Theta\). They therefore contribute exactly zero persistent
environment information and do not change the decision problem.

For independent persistent trivia \(D\), with reward depending only on the
base latent \(Z\),

\[
B_\rho^{(Z,D)}=B_\rho^Z.
\]

The upper bound ignores \(D\). For the lower bound, average any augmented
channel over \(D\), preserve its expected reward, and apply data processing.

## Proposition 8: public reward controls

If a public state-independent action can attain every finite threshold, then
PUBLIC-U has

\[
B_\rho=0
\]

at every finite threshold. This is attained by a finite constant action and is
not a limiting rare-burst argument.

For PUBLIC-C, let a fixed finite public action set attain bounded maximum
reward \(G_{\max}\). Then

\[
B_\rho^{\mathrm{PUBLIC-C}}
=B_{\rho-G_{\max}}^{\mathrm{base}}.
\]

This identity includes exact zero-information truncation and infeasibility
above the shifted base maximum. See
[Symbolic controls](symbolic-controls.md) for runtime and finite-projection
contracts.

## Numerical contract

The implementation must:

- use natural logarithms;
- reject invalid or zero-margin primary reward specifications;
- return zero for nonpositive feasible thresholds;
- return \(+\infty\) above attainable reward;
- preserve monotonicity and convexity within numerical tolerance;
- and reproduce the closed-form baseline to at least \(10^{-10}\).
