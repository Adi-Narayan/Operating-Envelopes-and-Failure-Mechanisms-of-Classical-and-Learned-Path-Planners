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

## Next
Day 1 per `instructions.md` Section 5: `prereg.md` for B3 (commit before launching), B1 and B2 diagnostics, C1 stall classification, WS-A skeleton; background C3 theta sweep; then B3 short runs.
