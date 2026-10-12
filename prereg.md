# Pre-registration

Each entry is committed before the runs it governs start. Selection rules are followed even when
the outcome is disappointing. Seeds are listed in `notes/seed_map.md`.

---

## WS-B, B3: short ablation of TD3 collapse (registered 2026-10-12, before any B3 run)

**Question.** Which single change most reduces collapse of TD3 trained with moving obstacles?

**Common setting.** `experiments/train_arc_seed.py --algo td3 --dynamic`, the paper's TD3
configuration (`experiments/baselines.py`, `BASELINE_TRAIN_CONFIG`), 50 episodes, CPU, one thread
per process. Learner seeds **10, 11, 12** for every condition (fresh seeds: not used by any
existing run, so no condition's outcome is known in advance). Outputs:
`results/v2/ws_b/b3/<condition>_seed<k>/`.

| Condition | Change from the paper's TD3 | Command flags |
|---|---|---|
| (a) baseline | none | (none) |
| (b) discount | gamma = 0.9 (ARC's value) instead of 0.99 | `--gamma 0.9` |
| (c) pre-tanh penalty | actor loss + lambda * E[u^2], **lambda = 2.0e-3** | `--pretanh-penalty 0.002` |
| (d) exploration | Gaussian exploration sigma = 0.3 of the action range instead of 0.1 | `--exploration-noise 0.3` |

**lambda (condition c)** was fixed from a separate baseline run (seed 100, 10 episodes,
`results/v2/ws_b/lambda_calibration/`): at episode 10 the actor loss was -3.07 and E[u^2] = 15.3,
so lambda = 0.01 * 3.07 / 15.3 = 2.0e-3 (penalty about 1% of the actor-loss magnitude).

**Outcome measure.** Collapse status of the **episode-50 checkpoint** (`arc_checkpoint_ep50.pth`)
of each run, evaluated exactly as `experiments/collapse_diagnostic.py`: one deterministic run on
the static custom environment from q0 heading towards the goal; collapsed if at least 99% of the
executed heading changes are at the 9 deg bound.

**Proxy validity (checked before registering).** On the existing runs, collapse status at the
episode-50 checkpoint matched the final status in 9/10 TD3 moving seeds and 10/10 TD3 static seeds
(`results/v2/ws_b/proxy_check/`), meeting the 9-of-10 criterion, so B3 stays at 50 episodes.

**Selection rule for B4.**
1. Count collapsed seeds (of 3) per condition.
2. The candidate is the condition among (b), (c), (d) with the fewest collapses.
3. Tie among (b), (c), (d): take the smallest change from published TD3, in this pre-set order
   (smallest first): (d) exploration noise, (b) discount, (c) added loss term. Rationale: (d)
   changes a tuning parameter that TD3 implementations routinely vary; (b) changes the
   objective's horizon; (c) adds a term to the loss.
4. B4 runs only if the candidate has **strictly fewer** collapses than (a). Otherwise B4 is
   skipped and the diagnosis (B1-B3) is reported alone.

## WS-B, B4: full fixed-TD3 run (conditional on B3)

The condition selected above, 10 learner seeds **0-9**, 300 episodes, moving obstacles,
`results/v2/ws_b/b4/td3fix_seed<k>/`. Same seeds as the paper's TD3 (moving) runs, so the
comparison with them is paired by seed.

## WS-B, B5: static SAC

SAC with the paper's configuration, trained on the static custom environment (no `--dynamic`),
10 learner seeds **0-9**, 300 episodes, `results/v2/ws_b/b5/sacstatic_seed<k>/`.

## Evaluation of B4 and B5 policies (fixed now)

The paper's protocol and scripts, unchanged:
- checkpoints per seed: the reward-retained checkpoint (`arc_best`, primary), episode 300, and the
  best checkpoint selected on the static training layout from 10 random headings
  (`eval_checkpoints.py` + the `select_oracle.py` rule);
- collapse status (`collapse_diagnostic.py` rule) of the retained checkpoint;
- every axis: static custom environment, the 50 paired dynamic environments, speed sweep, the
  three noise levels, 100 random layouts (static and dynamic), second layout (static and dynamic),
  with the frozen seeds of `notes/seed_map.md`.
No evaluation result is used to change any B4/B5 setting.

---

## WS-C, C4: damped APF (APF-D) selection (registered 2026-10-12, before any C4 tuning run)

**Candidates** (`experiments/apf_damped.py`; identical to the paper's APF when damping is off,
checked path for path in `results/v2/ws_c/c4_identity/`), all with full sensing, 1.8 m steps at
10 Hz, the paper's gains and theta = 25 deg:

| Name | Change |
|---|---|
| apf | none (plain APF, the reference for the constraint) |
| osc10 | oscillation-triggered 54 deg/s cap for K = 10 steps (trigger: heading change > 150 deg on two consecutive steps) |
| osc30 | as osc10 with K = 30 |
| mom050 | momentum, beta = 0.5 |
| mom070 | momentum, beta = 0.7 |
| mom085 | momentum, beta = 0.85 |
| cap | the existing always-on 54 deg/s cap |

**Tuning data (held-out only; see `notes/seed_map.md`).**
- 50 tuning layouts, static: `make_layout(500000 + j)`, planner seed 0.
- 50 tuning layouts, dynamic: same layouts, motion seed 510000 + j, planner seed 520000 + j (reported; not used by the rule).
- 50 held-out custom-environment motion seeds 500000-500049, planner seed 501000 + s, 18 m/s.

**Rule.**
1. Feasible: at most 1 dynamic success lost against `apf` on the 50 held-out motion seeds, counted
   per environment: environments that `apf` solves and the candidate does not (clarified
   2026-10-12 before any C4 result was inspected).
2. Among feasible candidates (including `apf` itself), take the one with the most successes on the
   50 static tuning layouts.
3. Ties: the smaller change, in this pre-set order (smallest first): apf, osc10, osc30, mom050,
   mom070, mom085, cap. (The oscillation-triggered cap acts only after a detected oscillation;
   momentum acts at every step, more so with larger beta; the always-on cap constrains every step.)
4. If the rule selects `apf`, no damped variant is carried forward and that is reported.

The selected variant is then evaluated once on every axis with the frozen evaluation sets.

---

## WS-D: envelope-derived arbiter, D1-D4 (registered 2026-10-12, before thresholds are computed or any tuning run)

**Planners** (`experiments/arbiter.py`; with switching disabled it reproduces each component path for
path, `results/v2/ws_d/identity/`).
- Default L: SAC trained with moving obstacles, the `arc_best` checkpoint of each of the 10 seeds;
  every result is reported per seed.
- D1 (matched information): C = APF under ARC's nine-ray sensing, 3.0 m steps at 6 Hz, no turn cap.
  If the C4 rule (above) selects a damped variant before D3 starts, C uses that variant's damping
  (momentum beta or oscillation-triggered cap K, with the cap scaled to 9 deg per 3.0 m step), and
  this file records which.
- D2 (privileged information, P1): C = DWA-CV (exact geometry and velocity estimates). It uses the
  thresholds selected for D1 (no separate tuning) and is labelled privileged wherever it appears.

**Signals and switching** exactly as in `experiments/arbiter.py`: d_min = minimum ray range; v_c =
closing speed of the minimum-range ray over the last two learned observations divided by the agent
speed, (d_prev - d_now) / 3.0 m; L -> C when d_min < d_on or v_c > rho_on; C -> L when d_min > d_off,
v_c < rho_off and C has been active for at least 1 s, or when C's goal progress over its last 20
decisions is below 10% of the distance travelled in them. d_off = d_on + 5 m, rho_off = 0.8 rho_on.

**D2, initial thresholds (deviation from instructions.md, recorded here).** instructions.md takes
the initial thresholds from SAC's existing dynamic-suite logs, which are evaluation data. To keep
every design choice off the evaluation sets they come instead from SAC alone (10 seeds) on the 50
held-out motion seeds 500000-500049:
- for every SAC collision, the learned observation 12 steps (2 s) before the collision;
- d_on = the median, over collisions, of d_min at that observation;
- rho_on = the median, over collisions, of the largest v_c over the 12 observations before the collision.
The same quantities over successful runs are reported (false-alarm view), not used.

**D3, tuning.** Grid: d_on in {d_on - 5 m, d_on, d_on + 5 m} x rho_on in {0.75, 1.0, 1.25} x rho_on,
with d_off and rho_off following the rule above. Held-out sets only:
1. the 50 held-out custom motion seeds 500000-500049 (18 m/s);
2. the 50 tuning layouts, static (`make_layout(500000 + j)`);
3. the 50 tuning layouts, dynamic (motion seed 510000 + j);
4. the held-out seeds with noise 2.5 m / 5 deg (noise seed 502000 + s);
5. the held-out seeds at r = 1.0 (agent speed 4.75 m/s).
APF's planner-perturbation seed: 501000 + s (sets 1, 4, 5), 0 (set 2), 520000 + j (set 3).
Objective: mean success over the five sets and the 10 SAC seeds. Ties: fewer switches per run.

**D4, evaluation (once, on every axis, frozen sets)**: per SAC seed, the arbiter against SAC alone
(same seed), C alone, and the oracle reference (an environment counts as a success if either
component succeeds), plus the fraction of time in C, switches per run, chattering (switches less
than 1 s apart) and per-step compute. If the arbiter does not beat its components, that is reported
with the oracle headroom and where it failed.

### Outcome of the C4 rule (recorded 2026-10-12, before D3 starts)
The rule selected **mom085** (momentum, beta = 0.85): 47/50 static tuning layouts against 30/50 for
plain APF, 0 held-out dynamic successes lost (50/50). It is the largest beta in the pre-registered
grid. Per the WS-D entry, the D1 fallback C is therefore APF under ARC's nine rays at 6 Hz with
momentum beta = 0.85 (`results/v2/ws_d/c_config.json`).
