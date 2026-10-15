# Scale-invariance audit (instructions.md Section 3.4)

Question: do outcomes depend on the agent speed v and the obstacle speeds only through the
ratio r = v_obs,max / v? `harness.set_ground_speed(v)` sets the APF decision period to
1.8 m / v and the obstacle integration tick to one third of it, so every agent step spans a
fixed number of obstacle ticks (APF and DWA: 3; learned planners: 5).

## Quantities with a time unit, and how each enters

| Quantity | Where | Unit as coded | Per agent step | Depends on speeds only through r? |
|---|---|---|---|---|
| Obstacle integration tick | `harness.TIMELINE_DT = APF_DT / 3` | s, scales with 1/v | fixed tick count | yes |
| Obstacle speed | `apf.assign_velocities` | m/s, absolute | displacement per tick = v_obs * 0.6 / v = 0.6 r (for the fastest type) | yes |
| Goal push-away | `apf.DynObstacle.step` | positional push 0.6 (50 - d + 10) m **per tick** | fixed (tick count per step is fixed) | yes, because the tick count per agent step is fixed; in absolute time the push rate scales with v |
| Wall reflection, obstacle-obstacle separation and velocity exchange | `apf.DynObstacle.step`, `handle_obs_collision` | positions; speeds restored to pre-contact magnitude | position-based | yes |
| APF step, stuck jitter, escape step, push-out | `apf.simulate_dynamic_apf`, `apf_variants.run` | m per decision (1.8, 0.35, 1.8 x 1.8, 3 m beyond margin) | fixed | yes |
| APF time budget | `harness.APF_MAX_STEPS` | 6000 decisions | fixed | yes (the paper's "600 s" holds at 18 m/s only) |
| Learned planners: step, turn limit, history, budget | `harness.ARCPlanner` | 3.0 m, 9 deg, 5 observations, 3600 steps per decision | fixed | yes |
| DWA turn rate, accelerations, horizon | `dwa.run` | rad/s, m/s^2, rad/s^2, s; all scaled with v (w ~ v, a ~ v^2, alpha ~ v^2, horizon 30 decisions) | turning radius 11.5 m, braking distance 18 m, look-ahead 54 m at every v | yes, by construction |
| DWA-CV velocity estimate | `dwa.run` | displacement over the last decision / decision period | prediction uses displacement per decision | yes |
| DWA time budget | `dwa.run(max_periods)` | 6000 decisions | fixed | yes |
| Sensing noise | `robustness_suite.py`, `dwa_eval.py` | m and deg **per decision** (one draw per obstacle or ray per decision) | fixed | yes |
| Wall-clock timing | `harness._summ` | ms | measurement only | not an outcome |

Not part of the speed sweep: moving-obstacle training (always at 18 m/s, r = 0.26) and the 3D worlds.

No quantity carries absolute time in a way that is not tied to the decision period, so in exact
arithmetic every outcome depends on speeds only through r.

## Bit-identity test

`scripts/verify_section3.py scaleinv` (results in `results/v2/verification/scale_invariance/`):
motion seeds 0-9, (A) agent at 18 m/s with baseline obstacle speeds, (B) agent at 9 m/s with every
obstacle velocity multiplied by 0.5 (same random draws), so r is equal.

| Planner | Same outcome | Same step count | Max pointwise path difference | Bit-identical runs |
|---|---|---|---|---|
| SAC (`sacdyn_seed0/arc_best`) | 10/10 | 10/10 | 0 m | 10/10 |
| APF | 10/10 | 10/10 | 4.8 cm (env 9), 4.2 mm (env 5); 1e-10 to 2e-9 m in the other 8 | 0/10 |

The APF differences are floating-point, not a time dependence (`scripts/verify_scaleinv_detail.py`,
`detail.json`): the two obstacle timelines differ by at most 2.3e-8 m (rounding in velocity x tick,
accumulated over thousands of ticks); APF's field amplifies this to millimetres on sensitive stretches
(env 9: above 1e-6 m from step 437 of 955) and the paths reconverge (final difference 1.4 mm); no escape
or push-out step occurs in either run. The learned policy is bit-identical because its observation is
float32, which absorbs differences of order 1e-8 m.

**Verdict: passed.** Outcomes depend on speeds only through r, exactly in exact arithmetic and to
floating-point tolerance in the implementation. This supports the proposition planned for WS-E (E1).
