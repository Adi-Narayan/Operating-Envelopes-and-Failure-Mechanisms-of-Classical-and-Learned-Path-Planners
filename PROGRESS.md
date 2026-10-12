# PROGRESS

Plan: `instructions.md` (kept outside the repository, in the paper folder; not committed).
Paper source: `paper.tex` in the paper folder (outside the repository).

## Status
- **Section 3 (verification pass): complete, all checks passed.** Nothing blocks the workstreams.
- No `[ASK ADI]` items open.
- Job queue: empty. No training or evaluation running.

## Log

### 2026-10-12: Section 3 verification pass
Code at commit `1e801cf` (HEAD before this block). Machine idle (no training) throughout; one thread per process.

**3.1 Reproduce before extending** (`scripts/verify_section3.py reproduce` -> `results/v2/verification/reproduce/`)
| Check | Paper | Reproduced | Run-by-run against stored results |
|---|---|---|---|
| APF, 50 dynamic environments, 18 m/s | 50/50, mean eta 72.7% | 50/50, 72.74% | 50/50 identical (path length difference 0.0) |
| SAC moving seed 0 (`sacdyn_seed0/arc_best`), dynamic suite | 40 (per-seed report) | 40 | 50/50 identical |
| APF, 100 random static layouts | 68/100, 32 timeouts | 68/100, 32 timeouts | 100/100 identical |
| DWA-CV, dynamic suite | 47/50 | 47/50 | 50/50 identical |

**3.2 Paper and code consistency**
- Section 4.3 "Only the learning algorithm changes" was wrong. Code (`experiments/baselines.py` `BASELINE_TRAIN_CONFIG`, `arc_2d/code/config.py`, `arc_2d/code/networks.py`) shows TD3/SAC differ from ARC in: optimiser and learning rate (Adam 3e-4 vs RMSprop 1e-3), discount (0.99 vs 0.9), soft-update rate (0.005 vs 0.01), batch (64 vs 32), replay capacity (50,000 vs 4,000), TD3 exploration noise (0.1 vs 0.2 of the action range; SAC samples), and encoder sharing (one encoder copy per actor and critic vs ARC's single shared LSTM). Gradient-norm clipping at 1.0 is **common to all three** (ARC clips too), so the paper's statement that clipping was "for CPU training" was also wrong. **Fixed in paper.tex** (Section 4.3, Table C.2 caption, Limitations).
- Collapse definition: `experiments/collapse_diagnostic.py` matches Section 4.3: retained checkpoint (`arc_best`), deterministic run on the static custom environment from q0 heading towards the goal, fraction of executed heading changes within 0.05 deg of the 9 deg bound, collapsed if >= 0.99.
- `arc_best` selection: the same rule for ARC, TD3 and SAC (`arc_2d/code/train.py`: highest mean training reward over the last 20 episodes, tracked from episode index 20, i.e. after 21 episodes; TD3/SAC reuse `ARCTrainer` with the agent swapped in `experiments/train_arc_seed.py`).

**3.3 Seed and layout map**: `notes/seed_map.md`.
- DWA tuning seeds found: motion 1000-1019 on the custom environment + the static custom environment (`experiments/dwa_tune.py`).
- New held-out sets defined, no collisions: motion 500000-500049 (APF perturbation 501000-501049, noise 502000-502049); tuning layouts `make_layout(500000-500049)` with motion 510000-510049 and dynamic APF perturbation 520000-520049. All 50 tuning layouts generate; none duplicates an evaluation layout.

**3.4 Scale-invariance audit**: `notes/scale_invariance_audit.md`; `scripts/verify_section3.py scaleinv`, `scripts/verify_scaleinv_detail.py` -> `results/v2/verification/scale_invariance/`.
- Every time-carrying quantity is tied to the decision period (obstacle tick = APF period / 3; goal push-away per tick; DWA limits scale with v; budgets and noise per decision).
- Bit-identity test (motion seeds 0-9; 18 m/s vs 9 m/s with obstacle speeds x0.5): SAC bit-identical 10/10; APF same outcome and step count 10/10, max pointwise difference 4.8 cm, traced to floating-point rounding in obstacle positions (<= 2.3e-8 m) amplified by the field, not to a time dependence. **Passed.**

### 2026-10-12 evening: Day 1 work blocks
**Infrastructure.** `scripts/jobqueue.py` (resumable queue: `.done` markers in `results/v2/queue/<name>/`, RAM gate, below-normal priority for evaluation), `scripts/v2common.py` (manifests). Named `jobqueue.py`, not `queue.py`, because a `queue.py` in `scripts/` shadows Python's `queue` module and breaks torch imports.
- Free RAM is the binding constraint (1.2-2.5 GB free with Adi's applications open; a training process uses ~0.36 GB, an APF job ~0.08 GB). Training runs 4 at a time, not 20.

**WS-B code changes (opt-in; defaults unchanged).** `baselines.py`: `Actor.pre()`, TD3 `pretanh_penalty` (default 0), logging of E[u^2] and |u| > 3 (TD3, SAC). `train_arc_seed.py`: `--gamma`, `--exploration-noise`, `--pretanh-penalty`, `--out-root`, per-episode `train_log.jsonl`. **Identity check passed:** with default flags, TD3 moving seed 0 reproduces the stored run's episode rewards 1-3 exactly (`results/v2/ws_b/checks/`).

**B3 pre-requisites.** 50-episode proxy valid: collapse at the ep-50 checkpoint matches the final status in 9/10 TD3-moving and 10/10 TD3-static seeds (`results/v2/ws_b/proxy_check/`). lambda = 2.0e-3 from a 10-episode TD3 run, seed 100 (`results/v2/ws_b/lambda_calibration/`); the pre-tanh output is already saturated at episode 0 (95% of |u| > 3). `prereg.md` B3/B4/B5 committed (`39a16ca`) before any B3 run.

**Running.** B3 (12 jobs, `jobs/b3.json`), B1 (`jobs/b1.json`), C3 theta sweep (`jobs/c3.json`), C4 tuning (`jobs/c4.json`; prereg `f762095`).

**C1 done** (`results/v2/ws_c/c1_classify/`): of the 32 stalls, 27 period-2 orbits around stable nodes, 4 non-periodic orbits around stable foci, 1 arena-boundary trap (layout 99: pinned at x = 500 m, force points out of the arena; the paper's "all in open space" needs this exception). curl/div = tan(theta) at every equilibrium to 1e-9; derived in `notes/c2_proposition.md` (Lemma 1: exact for goal exponent n = 2). **[ASK ADI]** check the C2 math before it goes into the paper.

**C4.** `experiments/apf_damped.py`; reproduces plain APF and the always-on cap path for path (`results/v2/ws_c/c4_identity/`).

**WS-D.** `experiments/arbiter.py` (L = learned, C = APF with ARC's rays at 6 Hz, or DWA-CV). Reproduces SAC alone, ray-APF alone and DWA-CV alone path for path on 10 environments (`results/v2/ws_d/identity/`).

## Next
- Pre-register D1-D3 (D2 thresholds from SAC on the held-out seeds, not from the evaluation logs: avoids tuning on evaluation data).
- When B3 finishes: apply the B4 rule; launch B4 + B5.
- When C4 finishes: apply the C4 rule (`scripts/ws_c_c4_select.py`).
- WS-A numbers pipeline.
