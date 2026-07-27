# Symbolic controls

ALEA, TRIVIA, PUBLIC-U, and PUBLIC-C are stationary controls for separating
reward-relevant information from novelty, acquired irrelevant information, and
public reward. All frontier values and information quantities are in nats.

## ALEA

`AleaRulebook` wraps any `RulebookRuntime` and preserves its latent, action
space, and reward. Each training observation also receives a cosmetic value
from a separate counter-based tape keyed by round, rule, query ordinal, and
channel. The required `cosmetic_seed` is sampled independently from the base
environment seed. Replaying a semantic event is deterministic for
reproducibility; different semantic events represent fresh draws.
Every wrapper exposes the same typed `observe_query` result, so ALEA can append
cosmetic novelty to useful or TRIVIA observations and an outer PUBLIC wrapper
can forward that combined observation without changing it.

The cosmetic tape is runtime randomness, not a persistent latent coordinate.
If \(U\) is its history, then

\[
P(U\mid\Theta)=P(U),\qquad I(\Theta;U)=0.
\]

ALEA is therefore absent from the finite decision problem. The
`alea_frontier_problem` helper returns the base problem unchanged.

## TRIVIA

`TriviaRulebook` wraps any `RulebookRuntime` and adds a disjoint namespace of
static labels
\(D_i\overset{\mathrm{iid}}{\sim}\mathrm{Uniform}[q]\). A `SymbolicQuery`
selects either a reward coordinate or a trivia coordinate. Both are observed
through the same q-ary channel family, but deployment reward depends only on
the reward namespace. The required `trivia_seed` is independently coupled to
the base latent seed. Useful and trivia labels are invariant to query order,
and their observation-noise keys are automatically namespace-separated.

For \(D\perp Z\), any augmented channel can be averaged over \(D\) to produce a
base action channel with the same expected reward. Data processing gives

\[
I(Z;A)\le I(Z,D;A).
\]

Conversely, a base channel can ignore \(D\). Hence, at every threshold,

\[
B_\rho^{(Z,D)}=B_\rho^Z.
\]

`enumerate_trivia_rulebook` exhaustively includes both useful and trivia
coordinates in the finite latent state while retaining the canonical
reward-relevant deployment actions. The generic solver remains free to choose
a channel separately for every full state, so this is the unrestricted
augmented optimization rather than a restricted policy test.
`augment_with_independent_trivia` applies the same exact source transform to
any existing `FiniteDecisionProblem`, including RED-C and MIX projections.

## PUBLIC-U

`UnboundedPublicRulebook` wraps any rulebook, pairs its deployment with a
finite nonnegative public integer \(k\), and adds the known reward
\(g(k)=k\,u_{\mathrm{public}}\).
For every finite \(\rho\), a finite constant choice of \(k\) attains the
threshold without depending on \(\Theta\). Thus

\[
B_\rho^{\mathrm{PUBLIC-U}}=0
\]

for every finite threshold. An infinite threshold is infeasible because every
individual action has finite reward. `public_u_witness` constructs the exact
constant-action witness; `public_u_bit_equivalent` is the analytic frontier.
PUBLIC-U is not finitely enumerated because any finite truncation would change
the global theorem.

## PUBLIC-C

`CappedPublicRulebook` wraps any rulebook and uses a fixed
`PublicBonusSchedule`. The schedule is a nonempty finite tuple of finite
nonnegative rewards and therefore contains an attained maximum \(G_{\max}\).
Its public choice is independent of the hidden latent and can be paired with
every base deployment.

Choosing a maximizing public action gives achievability. Conversely, removing
the public choice from any joint action loses no more than \(G_{\max}\) reward
and cannot increase mutual information. Therefore

\[
B_\rho^{\mathrm{PUBLIC-C}}
=B_{\rho-G_{\max}}^{\mathrm{base}}.
\]

The base frontier convention supplies the truncation: the transformed
frontier is zero through the shifted zero-information threshold and infinite
above the shifted maximum reward. `PublicCFrontier` and
`public_c_bit_equivalent` implement this identity.
`enumerate_public_c_rulebook` exhaustively crosses a finite base projection
with every bounded public choice for exact solver regression.
`augment_with_public_c` applies the same action/reward transform to any finite
base problem, so the registered controls compose over IND, RED-C, and MIX.

## Allocation safety

Finite control transforms use saturating exponentiation and multiplication to
check state count, action count, and reward-matrix entries before constructing
products, actions, reward rows, or enormous intermediate integers. They raise
if `max_matrix_entries` would be exceeded. PUBLIC-U is routed to its analytic
helper instead of allocating an artificial finite truncation.
