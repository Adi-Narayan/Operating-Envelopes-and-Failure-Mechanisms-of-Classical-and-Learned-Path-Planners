# C2: why APF stalls in free space (draft for Adi's check) [ASK ADI]

Numbers from `results/v2/ws_c/c1_classify/` (32 stall layouts of the 100 random static layouts).

## Setting

Field (Eqs. 1-3 of the paper): F(q) = -k_a (q - g) + R(theta) B(q), with
B(q) = sum_i k_r c_r rho_i(q)^-2 D(q)^n grad rho_i(q), D = |q - g|, n = 2.
Since grad(-1/rho) = rho^-2 grad rho, B = D^n grad Phi with Phi(q) = -sum_i k_r c_r / rho_i(q)
(ignoring the 0.8 m clamp, which is never active at these points: the equilibria are 16-39 m from
the nearest obstacle).

Motion (Eq. 4): q_{t+1} = q_t + Ds F(q_t) / |F(q_t)| with Ds = 1.8 m (the bias b is negligible:
|F_att| = 881-2775 at the equilibria, |b| ~ 0.06).

## Lemma 1 (structure of the Jacobian at an equilibrium)

At any equilibrium q* (F(q*) = 0), with P = (q*-g)(q*-g)^T / D^2 and S = Hess Phi(q*) (symmetric):

  J = DF(q*) = (-k_a I + n k_a P) + D^n R(theta) S.

Proof sketch. DB = D^n S + grad Phi (grad D^n)^T = D^n S + n D^-2 B (q*-g)^T. At the equilibrium
R B = k_a (q*-g), so B = k_a R^T (q*-g) and R DB = D^n R S + n k_a P. Add D F_att = -k_a I.

**Corollary 1 (n = 2).** -k_a I + 2 k_a P is symmetric with trace -2k_a + 2k_a = 0. Hence
div F(q*) = D^2 tr(R S) = D^2 cos(theta) tr S and curl F(q*) = D^2 sin(theta) tr S, so

  curl / div = tan(theta) at every equilibrium.

Check: all 31 interior equilibria give curl/div = 0.4663077, tan 25 deg = 0.4663077 (agreement to 1e-9).

**Corollary 2.** At theta = 0, J is symmetric, so its eigenvalues are real: equilibria are nodes
or saddles, never foci. The 4 stable foci found at theta = 25 deg exist only because of the rotation
(R S is not symmetric), and the rotation term is what can make the eigenvalues complex.

## Proposition 1 (fixed step cannot converge; period-2 orbit at a stable node)

Let q* be a stable node and consider motion along an eigendirection v, F(q* + x v) ~ lambda x v
with lambda < 0. Under Eq. 4 the iterate on this line is x_{t+1} = x_t - Ds sign(x_t). For any
x_0 in (0, Ds): x_1 = x_0 - Ds in (-Ds, 0), x_2 = x_0. So the iterate never converges; it settles on
the period-2 orbit {x_0, x_0 - Ds}, whose midpoint x_0 - Ds/2 lies within Ds/2 = 0.9 m of q*, and
the heading flips by 180 deg every step.

Check: 27 of the 32 stalls are period-2 orbits; every one surrounds a stable node; the midpoint is
0.08-0.85 m from the equilibrium (all < 0.9 m); the heading flips by 180 deg on each step.

**Stable foci (4 stalls).** With complex eigenvalues the force direction rotates around q*, so the
normalised fixed step does not return to the same two points: the orbit circles q* without a short
period (no period <= 12 found), staying within about one step of it.

## Proposition 2 (a heading-rate limit removes the flip)

If the heading may change by at most c < 180 deg per step, the reversal x -> x - Ds sign(x) is not
available: near q*, where the force direction flips, the agent turns at the maximum rate and moves
along an arc of radius Ds / (2 sin(c/2)) (19.1 m for c = 5.4 deg per 1.8 m step), i.e. it leaves
the 1-step neighbourhood of q*. Whether it then escapes depends on the field outside it; because
the basin is shallow (at every stall point the resultant was at most 36% of the attraction;
paper Section 6.7), the arc typically carries the agent out.

Check (existing result, paper Section 6.7): with the cap, timeouts on the 100 static layouts fall
from 29-40 to 3-13 in every sensing condition, while collisions rise from 0-3 to 4-8.

## The non-2-cycle stalls

- 4 stable foci (layouts 26, 34, 36, 97): Corollary 2 and the remark after Proposition 1.
- 1 arena-boundary trap (layout 99): the agent is pinned at x = 500 m; the force there points out of
  the arena (|F| = 394, almost entirely +x) and every step is clipped back to the wall. Not a free-space
  equilibrium; the paper's "stalls in open space" needs this exception.

## Terminology

With theta != 0 the field is not a gradient field (curl != 0 at every equilibrium), so these points
are stable equilibria (sinks) of the force field, not local minima of a potential. The paper should
say "stable equilibria (27 nodes, 4 foci)" instead of "local minima".

## Open for Adi
1. Check Lemma 1 and both corollaries.
2. Proposition 1 is exact on the eigenline; in 2D it is the 1D statement along the dominant direction.
   Is that rigour enough for an appendix "proposition with proof sketch"?
3. Proposition 2 is a mechanism statement, not a guarantee of escape. Agree with that framing?


---
## Revision after Adi's review (2026-10-13)

**Terminology (corrected).** "With theta != 0 the field is not a gradient field" was wrong by implication: the
simplified goal factor makes the repulsive term D^n grad(Phi), whose curl is grad(D^n) x grad(Phi), nonzero
wherever the two gradients are not parallel, at any theta. At an equilibrium they are parallel, so there the curl
comes from the rotation alone. Unlike Ge and Cui's potential-based form, the field has no potential at any theta.
Paper: Sections 2.1 and 4.1 and the stall section now say so.

**Corollary 1 scope.** Needs n = 2 AND d = 2 (in 3D the symmetric part has trace -k_a). Paper: "in the plane".

**Corollary 3 (new, exact).** tr S < 0 away from obstacles and region boundaries: Laplacian of -1/rho is -2/rho^3
(edge), -1/rho^3 (vertex), rho^-2 (1/(rho+a) - 2/rho) (circle of radius a). So div F(q*) = D^2 cos(theta) tr S < 0
for theta < 90 deg: no sources; sink iff det J > 0, saddle iff det J < 0. Check (`scripts/ws_c_c2_followup.py`):
divergence negative and det J > 0 at all 31 interior equilibria, i.e. all sinks.

**Bias.** The old argument compared |F_att| with |b|. Correct: b is constant within a run, so the motion's fixed
point is the zero of F + b. For planner seed 0, |b| = 0.011 (not 0.06-0.08 as assumed); the shift is 0.1-2.4 mm
(first-order |J^-1 b| agrees). Midpoints re-measured from the shifted point: 0.08-0.85 m (unchanged).

**Proposition 1, now exact for the linearised 2D map.** Every 2-cycle of T(p) = p + Ds Jp/|Jp| lies on a real
eigenline with negative eigenvalue (proof in the paper's Appendix F): foci have no 2-cycles. Transverse two-step
multiplier m = (1 - rho Ds/x0)(1 - rho Ds/(Ds - x0)), rho = |lam_t|/|lam_c|; symmetric cycle m = (1 - 2 rho)^2,
stable iff rho < 1, so stable cycles sit on the fast eigenline. Check on the 27 cycles: all within 3.3 deg of the
fast eigenvector, rho = 0.03-0.64, |m| < 1 for 26 of 27; the exception (layout 89, m = -1.39) has the most
asymmetric cycle (one point 0.13 m from the equilibrium), where the linearisation is least accurate. Asymmetry vs
anisotropy (Adi's prediction): midpoint distance vs rho, Spearman -0.32, p = 0.10 (right direction, not significant).

**Proposition 2 / one-mechanism hypothesis (tested, not supported).** The "36% resultant" figure is dropped.
APF-D is a soft turn limit: max heading change arcsin((1-beta)/beta) = 10.2 deg per step at beta = 0.85, turning
diameter 20.3 m, against 38.2 m for the 54 deg/s cap. Nearest saddle of F + b to each sink (Newton from a grid
within 80 m): on the 31 interior stalls the cap (full sensing, 10 Hz) removes 15, APF-D 20, both 14. APF-D removes
MORE with the SMALLER radius, so "a larger turning radius escapes more basins" does not hold. The fixed-diameter
rule predicts poorly (cap: 14 of 28 predicted escapes removed; APF-D: 8 of 10 predicted, but 12 of 21 predicted
non-escapes also removed). Nearest-saddle distance is a weak predictor: AUC 0.72 (p = 0.04, unadjusted) for the
cap, 0.65 (p = 0.19) for APF-D. Paper states this as measured.

**[ASK ADI]** remaining: (1) Proposition 1 (exact) and Corollary 3 as now written in Appendix F; (2) the framing
of the failed one-mechanism test in the stall section.
