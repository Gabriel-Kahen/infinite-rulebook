# Finite reward-information solver contract

This note defines the numerical problem solved by the finite frontier stack.

## Problem

For finite environment states \(\theta\in\{1,\ldots,n\}\), actions
\(a\in\{1,\ldots,m\}\), prior \(p_\theta\), and reward matrix
\(r_{\theta a}\), solve

\[
B_\rho
=
\min_W I_p(W)
\quad\text{subject to}\quad
R_p(W)\ge\rho,
\]

where each row of \(W_{a\mid\theta}\) is a probability distribution,

\[
R_p(W)
=
\sum_{\theta,a}p_\theta W_{a\mid\theta}r_{\theta a},
\]

and

\[
I_p(W)
=
\sum_{\theta,a}
p_\theta W_{a\mid\theta}
\log\frac{W_{a\mid\theta}}{q_a},
\qquad
q_a=\sum_\theta p_\theta W_{a\mid\theta}.
\]

All calculations use natural logarithms.

## Feasibility endpoints

The best zero-information reward is

\[
R_0=\max_a\sum_\theta p_\theta r_{\theta a}.
\]

Therefore \(B_\rho=0\) for \(\rho\le R_0\).

The maximum attainable reward is

\[
R_{\max}
=
\sum_\theta p_\theta\max_a r_{\theta a}.
\]

Thresholds above \(R_{\max}\) are infeasible.

## Lagrangian problem

For \(\beta\ge0\), solve

\[
F_\beta^\star
=
\inf_W\left[I_p(W)-\beta R_p(W)\right].
\]

Given an action marginal \(q\), the row update is

\[
W_{a\mid\theta}
=
\frac{q_a\exp(\beta r_{\theta a})}
{\sum_bq_b\exp(\beta r_{\theta b})}.
\]

The marginal update is

\[
q_a\leftarrow\sum_\theta p_\theta W_{a\mid\theta}.
\]

The implementation performs normalizers and dual slacks in log space. It may
temporarily restrict optimization to actions with nonnegligible mass and KKT
slack, but it checks the certificate over the full action set and reintroduces
every excluded action that violates the global optimality conditions.

## Certified dual lower bound

For a candidate marginal \(q\), define

\[
f_\beta(q)
=
-\sum_\theta p_\theta
\log\sum_a q_a\exp(\beta r_{\theta a}).
\]

The Lagrangian optimum is

\[
F_\beta^\star=\min_{q\in\Delta_m}f_\beta(q).
\]

For every action, define the multiplicative BA residual

\[
s_a
=
\sum_\theta
p_\theta
\frac{\exp(\beta r_{\theta a})}
{\sum_bq_b\exp(\beta r_{\theta b})}.
\]

Let \(s_{\max}=\max_a s_a\). Scaling the dual variables by
\(s_{\max}^{-1}\) gives the certified bound

\[
F_\beta^\star
\ge
-\sum_\theta p_\theta\log Z_\theta-\log s_{\max},
\qquad
Z_\theta=\sum_aq_a\exp(\beta r_{\theta a}).
\]

Thus, for every target \(\rho\),

\[
B_\rho
\ge
\beta\rho
-\sum_\theta p_\theta\log Z_\theta
-\log s_{\max}.
\]

This bound remains valid before full convergence. Every action, including a
numerically zero-mass action, is included in \(s_{\max}\). The feasible
channel witness provides the upper bound.

## Target solving

The target solver:

1. handles \(R_0\) and \(R_{\max}\) explicitly;
2. brackets the target with Lagrangian solutions at two multipliers;
3. mixes the two conditional channels to hit the target reward;
4. recomputes the mixture mutual information directly;
5. maximizes valid dual lower bounds from all visited multipliers;
6. and reports the primal/dual interval and diagnostics.

The mixture channel is a feasible witness. Its mutual information is recomputed
after mixing and is no greater than the mixture of the endpoint information
values.

At \(R_{\max}\), finite multipliers are unnecessary and potentially
scale-dependent. The implementation instead restricts each state to its
reward-maximizing actions and directly solves the resulting
support-constrained minimum-information problem. This also minimizes
information correctly when maximizing-action sets overlap.

## Numerical acceptance

- Channel rows sum to one within \(10^{-12}\).
- Rewards and mutual information are independently recomputed from witnesses.
- Lower bounds never exceed feasible upper bounds beyond tolerance.
- Analytic and finite-solver frontiers agree on one-coordinate \(q=2\) and
  \(q=4\) problems.
- Bound width is below \(10^{-8}\) nats on registered small regression cases.
- Every returned upper bound retains its channel witness.
