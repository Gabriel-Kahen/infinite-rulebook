# Assumption stress tests

## How to read this file

Each row removes or weakens one assumption while leaving the others as close
as possible to the theorem under test. Outcomes are:

- **fails:** an explicit counterexample violates the desired conclusion;
- **ill-posed:** the reward, channel, or limiting objective need not exist;
- **survives:** the assumption is not needed for that restricted conclusion;
- **open/replaceable:** the written assumption is sufficient, but the exact
  weakest replacement has not been proved here.

All logarithms are natural. “Frontier” always means an infimum; a sequence
with information tending to zero need not supply an information-zero channel.

## Reward transformations

| Assumption removed or weakened | Outcome | Witness or argument | Consequence for the paper |
|---|---|---|---|
| Positive slope in R1 | **Fails.** | Let \(\Theta\) be a fair bit, \(A\in\{0,1\}\), and \(r(\theta,a)=\mathbf1\{a=\theta\}\). Under \(r'=-r\), target \(0\) requires being wrong almost surely and costs \(\log2\), while \(B_r(0)=0\). A negative slope reverses, rather than relabels, the upper reward constraint. | Keep \(\alpha>0\) explicit. |
| Nonzero slope in R1 | **Fails.** | If \(r'=\beta\) is constant, every distinction among original reward thresholds is erased. | Do not describe nonnegative-affine maps as safe; the slope must be strictly positive. |
| Global constants \(\alpha,\beta\) in R1 | **Fails.** | Let \(\Theta\) be a fair bit, \(A\in\{0,1\}\), and \(r=\mathbf1\{A=\Theta\}\). Replacing a global shift by the action-dependent term \(\beta(A)=\mathbf1\{A=0\}\) gives transformed reward \(r'=r+\mathbf1\{A=0\}\). The fixed action 0 has transformed mean \(3/2\), so \(B_{r'}(1)=0\), whereas \(B_r(1)=\log2\). | “Change of units” must be global, not state- or action-dependent. |
| Transform conditional mean versus transform noisy realized reward | **Fails.** | Let realized \(R\in\{-1,+1\}\) be fair conditional on the only state/action. Its conditional mean is zero. With the strictly increasing \(g(x)=e^x\), \(g(\mathbb E R)=1\) but \(\mathbb E g(R)=\cosh(1)\). | Define precisely which reward object is transformed. R1/R2 concern the mean-payoff function; R3 is deterministic so the distinction disappears. |
| Affinity, retaining only invertible monotonicity | **Fails dramatically.** | The prefix construction in R3 is open-ended under \(r\) but has \(B^{r^3}(\rho)=0\) at every finite threshold, although cubing is invertible and feedback is recoverable. | This is the principal counterexample; open-endedness is cardinal, not ordinal. |
| Universality over all finite problems in R2 | **Survives for some nonlinear maps on a fixed problem.** | If a nonlinear \(g\) agrees with \(\alpha x+\beta\) on one problem’s finite payoff support, it induces exactly the same transformed problem as that affine map. | R2 must say “for every finite decision problem”; it is not a problem-specific classification. |
| Universality over lotteries in R2 | **Fails as a route to affinity.** | Constant one-action problems identify only \(h=g\). Without nondegenerate lotteries, they impose no Jensen equality and cannot distinguish an arbitrary increasing \(g\) from an affine one. | Retain finite source lotteries in the quantified problem class. |
| Continuity of \(g\) in R2 | **Open/replaceable.** | The proof needs enough regularity, together with Jensen equality, to exclude pathological additive solutions. Monotonicity itself often supplies that regularity, so explicit continuity may be redundant. | Present continuity as a clean sufficient assumption; list weakening it as an axiom-minimization question, not a finding. |
| Bijection of sequential reward histories in R1-S | **Fails.** | If the transformed environment clips, coarsens, delays, or adds noise to reward observations, an old policy may not be reconstructible even when the mean-payoff functions are affinely related. | Keep static affine invariance separate from sequential policy transport. |
| Unbounded reward/superlinear growth in R3 | **The specific collapse proof fails; bounded reward prevents positive-gap collapse.** | If transformed reward has oscillation \(L<\infty\), N1 gives \(B(\rho)\geq2((\rho-R_0)/L)^2>0\) for \(\rho>R_0\). The on/off witness needs \(g(n)/n\to\infty\). | Do not generalize R3 to bounded monotone transforms. |

## Source and action representation

| Assumption removed or weakened | Outcome | Witness or argument | Consequence for the paper |
|---|---|---|---|
| Exact reward factorization through \(S\) in S1 | **Fails.** | Let \(S\) be constant, \(\Theta=(S,U)\) with fair bit \(U\), actions \(A\in\{0,1\}\), and reward \(\mathbf1\{A=U\}\). The full-source channel \(A=U\) reaches reward one at cost \(\log2\). A reduced constant state with the conditional-mean payoff \(1/2\) cannot reach one. | Conditional averaging of reward is not a substitute for pointwise factorization. |
| Independence of discarded coordinate \(D\) from sufficient state \(Z\) | **Survives statically.** | If \(r((z,d),a)=\bar r(z,a)\), S1 applies to \(S=Z\) for every joint law of \((Z,D)\). | Do not add an unnecessary independence assumption to the static theorem. Correlated \(D\) may still be an acquisition proxy. |
| Availability of all randomized kernels / closure under fiber averaging in S1 | **Fails for constrained channel classes.** | Let \(S\) be constant, \(\Theta=(S,U)\) with fair \(U\), and reward identically zero. If the declared full-source admissible class contains only \(A=U\), its frontier at zero is \(\log2\); the natural reduced problem admits a constant action at cost zero. Fiber averaging was forbidden. | Any constrained extension must require both lifting and fiber averaging to preserve admissibility. |
| Exact reward preservation by action map \(q\) in A1 | **Fails.** | In the matching-bit problem, merging raw actions 0 and 1 into one quotient label cannot support a well-defined quotient reward: their reward profiles differ by state. Any chosen quotient payoff changes at least one raw behavior. | Quotient behavioral equivalence must include the complete payoff profile and any state-dependent feasibility/dynamics being priced. |
| Surjectivity onto admissible quotient actions | **Fails.** | Let the raw action set contain only \(a_0\) with reward 0, while the quotient model also contains \(\bar a_1\) with constant reward 1; map \(a_0\mapsto\bar a_0\). Then at \(\rho=1\), the quotient frontier is 0 and the raw frontier is \(+\infty\). | The reverse inequality cannot include quotient actions with no raw implementation. |
| Source-independent admissible lift in A1 | **Fails in constrained models.** | Let \(\Theta\) be a fair bit, the quotient have one action \(\bar a\) of reward 0, and its raw fiber be \(\{a_0,a_1\}\), also reward 0. Impose state-dependent feasibility that only \(a_\theta\) may be selected. The quotient constant costs 0 bits, but every admissible raw channel reveals \(\Theta\) and costs \(\log2\). No common representative is admissible in both states. | “Lift” must mean source-independent **and** feasible/resource-preserving, not merely a set-theoretic preimage. |
| Measurability of the lift outside finite spaces | **Open/replaceable.** | A set-theoretic section need not be an admissible Borel kernel in the chosen measurable category, and resource constraints can destroy closure. No general selection theorem is proved in the draft. | State the finite theorem cleanly; make any standard-Borel extension conditional on a measurable stochastic lift. |
| Blackwell dominance in A2 | **Fails.** | Let the payoff state be two independent fair bits \((X,Y)\); \(E_1\) reveals only \(X\), and \(E_2\) only \(Y\). If reward requires guessing \(X\), \(E_1\) reaches reward one while \(E_2\) cannot exceed \(1/2\); for guessing \(Y\), the ordering reverses. | Do not infer frontier order from an informal claim that one representation “contains more information”; require a garbling relation for the fixed experiment theorem. |

## Baseline, compactness, and tail control

| Assumption removed or weakened | Outcome | Witness or argument | Consequence for the paper |
|---|---|---|---|
| Strict gap \(\rho>R_0\) when claiming positive noncollapse | **Fails at the boundary.** | Let \(\Theta\) be a fair sign, actions \((n,s)\), \(n\geq3\), and \(r(\theta,(n,s))=1-1/n+\tfrac12\theta s\). Then \(R_0=1\) is not attained, but choosing \(P(s=\Theta)=1/2+1/n\) reaches reward 1 with information \(\log2-h(1/2+1/n)\to0\). Hence \(B(R_0)=0\) nonattained. | N1 is a strict-positive-gap theorem. Do not infer attainment or strict positivity at \(R_0\). |
| Distinction between frontier infimum and minimum | **Fails.** | The preceding bounded example has \(B(1)=0\), yet no information-zero channel attains reward 1. | Use “infimum” consistently and distinguish attained zero, boundary nonattainment, positive-gap collapse, and infeasibility. |
| Bounded oscillation in N1 | **Fails.** | In the strict-margin example, \(\Theta\) is uniform on \([q]\), action \((M,j)\) pays \(Mu\) if correct and \(-Mc\) otherwise with \(c>u/(q-1)\), and abstention pays 0. Thus \(R_0=0\), yet activating \((M,\Theta)\) with probability \(\rho/(Mu)\) gives reward \(\rho\) and information \(\rho\log q/(Mu)\to0\). | Some tail/integrability contract is genuinely necessary above baseline. |
| UI under \(P_n\), retaining no control under \(Q_n\) | **Fails even with product references.** | Let \(\Theta\) be a fair bit. With probability \(1/n\), output action \((n,\Theta)\), otherwise abstain, and set reward \(2n\mathbf1\{A=(n,s),s\ne\Theta\}\). Under the dependent law \(P_n\), reward is always zero and hence UI. Under \(Q_n=P_\Theta P_{A,n}\), mean reward is one and the tail does not vanish. TV and information tend to zero, but the expectation difference is \(-1\). | The symmetric convergence lemma cannot assume UI only under the induced joint laws. |
| UI under \(Q_n\), retaining no control under \(P_n\) | **Fails even with product references.** | Let \(\Theta\) be iid fair bits. With probability \(d_m=2^{-m}\), output the correct length-\(m\) prefix, otherwise abstain; reward is \(2^m\) for a correct prefix. Then \(I\leq2^{-m}m\log2\to0\), \(\mathbb E_{P_m}r=1\), and \(P_m\) is not UI. Under \(Q_m\), the prefix matches with probability \(2^{-m}\), so its tail expectation is \(2^{-m}\to0\); the \(Q_m\) family is UI. | The symmetric lemma also cannot assume UI only under independence references. |
| UI failure treated as sufficient for collapse | **Fails.** | Let \(\Theta\) be a fair sign, choose deterministic action \(A=n\), and set \(r(\theta,n)=n\theta\). Then \(P_n=Q_n\), information and the reward gap are exactly zero, but the rewards \(\pm n\) are not uniformly integrable. UI failure alone creates no dependent-versus-independent advantage. | Phrase N2 only as a necessary condition for positive-gap collapse. |
| Compact action space in N4 | **Fails for attainment.** | The bounded noncompact action example above has \(R_0=1\) unattained and \(B(R_0)=0\) only as an infimum. | Compactness is one clean way to close the boundary loophole. |
| Continuity/upper-semicontinuity of reward in N4 | **Fails even on a compact action space.** | Use a one-state problem with compact \(\mathcal A=\{0\}\cup\{1/n:n\geq1\}\), and \(r(1/n)=1-1/n\), \(r(0)=0\). Then \(R_0=1\) is not attained; threshold 1 is infeasible, so \(B(1)=+\infty\), contradicting the claimed equivalence at \(\rho=R_0\) if continuity is dropped. | Boundedness plus compactness is not enough; retain continuity or an appropriate upper-semicontinuity/closed-feasibility replacement. |
| Weak closedness of a restricted kernel class in N4 | **Fails for the compactness proof.** | Start with a compact finite problem and delete its information-minimizing kernel while retaining a sequence converging to it. The restricted feasible class is not closed, so the infimum need not be attained. | Any resource/support restriction needs an explicit closedness proof. |
| Polish/standard-Borel regularity in N4 | **Open/replaceable.** | The proof uses Prokhorov compactness and disintegration through joint laws. More general measurable spaces require different compactness machinery and may lack regular conditional kernels. | Do not state the topological theorem beyond its written category without a separate proof. |

## Composition and countable limits

| Assumption removed or weakened | Outcome | Witness or argument | Consequence for the paper |
|---|---|---|---|
| Independence of component sources in C1 | **Fails.** | Let \(\Theta_1=\Theta_2=Z\) be the same fair bit; each component rewards \(\mathbf1\{A_i=\Theta_i\}\). Reward 2 is achieved by \(A_1=A_2=Z\) at information \(\log2\), whereas the independent-component infimal convolution at target 2 is \(2\log2\). | Independence is essential to the superadditive information lower bound. |
| Additive reward in C1 | **Fails.** | Let \(\Theta_1,\Theta_2\) be independent fair bits and use joint reward \(\mathbf1\{A_1=\Theta_1\text{ and }A_2=\Theta_2\}\). Target 1 requires both bits and costs \(2\log2\). If one tried to allocate target 1 using the two matching-bit component frontiers, \(\rho_1=\rho_2=1/2\) would cost zero, so the infimal-convolution formula would be wrong. | Synergistic or substitutive rewards need a different multivariate frontier; scalar reward allocation is justified only by additivity. |
| Product action feasibility in C1 | **Fails on achievability.** | For two independent matching-bit components, restrict global actions to pairs satisfying \(A_1=A_2\). The component product channel that attains reward 2 is inadmissible, and reward 2 is impossible when the bits differ, although the unrestricted infimal convolution is finite. | State product actions/admissibility explicitly. |
| Identical components in C2 | **The equal-allocation formula fails, while C1 survives.** | If one component has a zero-information positive-reward action and another requires information for any positive gap, optimal allocations are unequal. | Use infimal convolution for heterogeneous components; reserve \(nB_1(\rho/n)\) for identical frontiers. |
| Convex randomized-channel frontier in C2 | **Fails for deterministic-only decisions.** | Deterministic channel classes are not closed under mixture, so the component cost curve can be nonconvex and equal allocation need not minimize its \(n\)-fold infimal convolution. | Randomized behavioral channels are part of the base definition, not a cosmetic choice. |
| iid source in C3 | **Fails as stated.** | Let every component source equal the same fair bit \(Z\). Each component has abstention and guesses: a correct guess pays \(+1\), an incorrect guess pays \(-c\), with \(c>1\). For sufficiently small \(\lambda>0\), \((e^\lambda+e^{-c\lambda})/2<1\); Donsker--Varadhan therefore gives \(B_1(x)\geq\lambda x\), so \(\kappa>0\). Globally, activate \(N\) correct copies with probability \(d=\rho/N\) and abstain otherwise. This earns \(\rho\), has information \(d\log2\to0\), and even has \(\mathbb EN(A)=\rho\). Thus \(B_\infty(\rho)=0\ne\kappa\rho\). | C3 is an iid theorem; dependent countable products require a dependence-sensitive formula. |
| Pointwise zero null action in C3, retaining only mean-zero | **Ill-posed.** | Let \(\Theta_i\) be iid fair signs and let the designated inactive action satisfy \(r(\Theta_i,a_0)=\Theta_i\). It has mean zero, but playing it on infinitely many “inactive” coordinates produces the divergent series \(\sum_i\Theta_i\), so the global reward is not defined. | The null action must be pointwise zero, or the infinite product needs an explicit centering/summation convention. |
| Finite support almost surely in C3 | **Open/replaceable, not safely removable.** | With infinitely many nonnull coordinates, even bounded component rewards can yield a divergent reward series. Absolute summability of realized rewards would be a possible replacement, but is not proved equivalent to the current action model. | Keep finite support for the stated theorem; study infinite-support actions separately. |
| Finite expected support in C3, with only finite support almost surely retained | **Ill-posed without a replacement integrability condition.** | In a one-state problem with nonnull actions paying \(+1\) or \(-1\), independently choose a finite block length \(N\) with probabilities proportional to \(1/n^2\) and an independent fair sign, then fill the block with that sign. Every action has finite support, but \(\mathbb E N=\infty\); positive and negative parts of total reward both have infinite expectation, so \(\mathbb E r_\infty\) is undefined. | Finite expected support is sufficient rather than logically unique. A direct requirement \(\mathbb E\sum_i|r_i|<\infty\) may replace it, but that extension should be stated and proved separately. |
| Absolute convergence/Fubini in the lower bound of C3 | **Fails as a proof step.** | Without absolute integrability, \(\mathbb E\sum_i r_i\) need not equal \(\sum_i\mathbb E r_i\); partial reward allocations can have no limit or a different conditional sum. | Record absolute convergence explicitly before taking \(m\to\infty\). |
| Ability to activate arbitrarily many finite coordinates in C3 | **Fails on the upper bound.** | Impose a hard support cap \(N(A)\leq M\). For targets above the maximum reward attainable on \(M\) coordinates, the frontier is \(+\infty\), whereas \(\kappa\rho\) is finite. | The local-slope upper bound depends on spreading reward over \(n\to\infty\) coordinates. |
| Positive local slope \(\kappa>0\) when claiming noncollapse | **Fails by definition when \(\kappa=0\).** | C3 itself yields \(B_\infty(\rho)=0\) for all finite \(\rho\) if \(B_1(x)=o(x)\) near zero. In particular, any positive zero-information component reward makes \(B_1\) vanish near zero. | Task count alone does not imply open-ended information demand; the local price is the controlling quantity. |

## Tests that should become executable checks

The counterexamples above are proofs, but a small exact/numerical suite can
guard the implementation and typesetting against regressions:

1. enumerate binary-source finite channels on a rational grid and compare R1
   before/after several affine maps;
2. verify the matching-bit nonlinear and negative-slope failures;
3. compare a full source to its exact sufficient statistic and to the
   deliberately invalid conditional-mean reduction;
4. verify quotient pushforward inequality and strict failure when an extra
   quotient action has no lift;
5. compute the Pinsker lower bound and the bounded boundary example as
   \(n\) grows;
6. plot information decay in the strict-margin, small-correlation, and cubed
   prefix constructions;
7. compute two-component independent and duplicate-source strict-margin
   frontiers by convex programming and compare them with infimal convolution;
   and
8. check \(nB_1(\rho/n)\to\kappa\rho\) for several finite component problems,
   while separately testing the hard-support-cap failure.

These checks are diagnostics, not evidence replacing the proofs. Any mismatch
should be treated first as a theorem-assumption or discretization bug, not as a
statistical experimental result.
