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
