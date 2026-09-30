# experiments/ — verification and re-run of the paper's results

Everything in the revised paper is produced by the scripts here, using the corrected evaluation protocol described in `FINDINGS.md` §0.
- `FINDINGS.md`: what was checked, what was found, and the evidence.
- `PAPER_EDITS.md`: replacement text for the paper.
- `out/report.md`: every table, generated.

Dependencies: Python 3.12, NumPy, PyTorch (CPU build is enough), Shapely, SciPy, Matplotlib, tqdm, TensorBoard (only for logging), openpyxl and psutil (used by the original scripts).

## Protocol (shared by both planners)
`harness.py` defines it:
- **Layout:** Corridor Gauntlet (identical geometry in `static_apf`, `dynamic_apf` and ARC's `corridor_gauntlet`).
- **Start / goal:** start (35, 35), goal (465, 465), 10 m goal radius.
- **Collision:** exact clearance < 5 m.
- **Budget:** 10.8 km travel.
- **Speed:** both planners at the same ground speed (default 18 m/s).
- **Obstacle motion:** one timeline per seed on a common grid, so APF and ARC see identical motion.

## Reproducing each result

| Result | Command | Output |
|---|---|---|
| Static APF checks (original-protocol reproduction, escape/snap mechanism, 0.8 m clamp, gain-ratio invariance, θ sweep), Table 6 | `python verify_apf_static.py` | `out/verify_apf_static.json` |
| (k_a, k_r) sweep and its wall time | `python gain_sweep.py` | `out/gain_sweep.json` |
| Table 7 (APF, paper's 3D environment) | `python run_apf3d.py` | `out/apf3d.json` |
| Table 7b (APF in the 3D ARC arena) | `python apf_in_arc3d.py` | `out/apf_in_arc3d.json` |
| Train one ARC seed (≈2 h on one CPU thread) | `python train_arc_seed.py --seed N [--env env_apf_urban]` | `out/arc_models/seedN/` or `out/arc_models/env_apf_urban_seedN/` |
| Evaluate every checkpoint on a static layout | `python eval_checkpoints.py --layout corridor --dirs out/arc_models/seedN` | `out/checkpoints_*.json` |
| Choose the upper-bound ("oracle") checkpoints | `python select_oracle.py` | `out/oracle_checkpoints.json` |
| Tables 3 and 4 and the transfer matrix | `python evaluate.py --layout {corridor,urban} --models <model list> --tag final_<layout>_a` | `out/final_*.json` |
| Speed-ratio envelope | `python speed_sweep.py --layout corridor --models out/arc_models/seed*/models/arc_best.pth` | `out/speed_sweep_corridor*.json` |
| Timing (run on AC power with nothing else running) | `python timing.py --models <successful ARC checkpoints>` | `out/timing.json` |
| Train a TD3 / SAC baseline (same encoder, observation, reward, budget) | `python train_arc_seed.py --algo {td3,sac} --seed N [--dynamic]` | `out/arc_models/{td3,sac}[dyn]_seedN/` |
| Queue many training runs (memory guard, machine-wide process cap) | `python run_queue.py --parallel P [--only sacdyn] [--global-cap 9] [--log-suffix _x]` | `out/logs/` |
| Monitor training (progress / health / pause to free RAM) | `python progress.py --watch 600`; `python watchdog.py`; `python pause_jobs.py {pause "<args>",resume,list}` | `out/logs/progress.txt` |
| APF with ARC's constraints (Table `tab:matched`) | `python test1_matched.py` | `out/test1_matched.json` |
| Sensing noise / 100 random layouts | `python robustness_suite.py --test {noise,layouts} --models <ckpts> --tag <name> [--no-apf] [--n-envs 10] [--n-layouts 10]` | `out/noise_*.json`, `out/layouts_*.json` |
| Training-collapse diagnostic | `python collapse_diagnostic.py` | `out/collapse.json` |
| Timing of the baselines | `python timing.py --no-3d --out timing_baselines --models <ckpts>` | `out/timing_baselines.json` |
| All tables (incl. learned baselines, seed-level tests, envelope summary) | `python report.py` | `out/report.md`, `out/report.json` |
| Figures | `python figures.py --tag final_corridor_a` | `out/figures/` |
| Supplementary: original model on its own layout | `python urban_head_to_head.py`; `python evaluate.py --layout urban [--speed 3] --models ../arc_2d/models_original/arc_best.pth` | `out/urban_*.json` |

The model list used for `final_*_a` is in `out/final_model_list_a.txt`. Seed 1 (which finished last) was evaluated separately as `final_*_b`, and `report.py` merges the two.

**Stored models.** For each retrained seed, `out/arc_models/<seed>/models/` keeps only the checkpoints the paper evaluates: `arc_best.pth`, `arc_checkpoint_ep300.pth` and the best-by-evaluation checkpoint named in `out/oracle_checkpoints.json`. The per-checkpoint results for every checkpoint (ep50–ep300) are preserved in `out/checkpoints_*.json`. Re-running `train_arc_seed.py` with the same seed regenerates all of them. All seven original checkpoints are in `arc_2d/models_original/`. For the TD3 and SAC runs the same three checkpoints are stored as actor-only files (`publish_models.py`, `baselines.export_actor`): evaluation uses only the deterministic actor, and each file was checked to reproduce the full checkpoint's actions and runs exactly. The full training checkpoints (critics, targets, optimisers; 1.6 GB) are not in the repository.

**Figures in the paper.** `figures.py` writes to `out/figures/`. The paper uses them under these names:
- `env2d_apf.png` ← `fig1_static_final_corridor_a.png`
- `env3d.png` ← `fig_env3d.png`
- `dyn_examples.png` ← `fig_dyn_examples.png` (panels: `fig3_shared_success_final_corridor_a.png`, `fig4_arc_collision_final_corridor_a.png`)
- `speed_envelope.png` ← `fig_speed_envelope.png`
- `arc_training.png` ← `fig5_arc_training.png`
- `theta_sweep.png` ← `fig6_theta_sweep.png`
- `apf_arc3d.png` ← `fig_apf_in_arc3d_stage2.png`
- `baselines_seeds.png` ← `fig_baselines_seeds.png`
- `baselines_training.png` ← `fig_baselines_training.png`

**Baseline evaluations.** TD3/SAC Table 3/4 files are `out/final_corridor_{td3dyn,td3,sacdyn}_{a,b,c}.json` and `out/final_urban_learned_{a,b}.json`; the non-learning TD3 seeds were spot-checked on 10 environments (`*_collapsed10.json`, `out/spot_urban_collapsed10.json`, named `spot_` so that `report.py` does not merge it with the 50-environment files). The one GPU-trained TD3 run (static seed 2) was replaced by a CPU retrain; it is not used and not published (its checkpoint results are in `out/gpu_run_td3_seed2_checkpoints.json`).

## Changes to the original code
- **`static_apf/static_apf.py`, `dynamic_apf/apf.py`, and the `core.py` heredoc in `apf_3d/create_project_unified.sh`:**
  - 10 m goal radius;
  - no appended snap-to-goal segment;
  - exact-clearance collision check (< 5 m);
  - escape / push-out / clearance diagnostics.
- **`dynamic_apf/apf.py` only:**
  - fixed seeds, 50 runs;
  - optional shared obstacle timeline.
- **`arc_2d/code/config.py`:** new `corridor_gauntlet` environment, an exact copy of the APF layout.

The planners themselves (forces, gains, θ, step size, escape rule, ARC network and training) are unchanged.

## Note on machine state
If a laptop, it must be on AC power with a performance power plan. Between 01:20 and 02:05 on 25 Sep the machine ran on battery and throttled to about 1.9 GHz. That inflated the corridor-seed training wall times, but it did not affect any result or timing measurement.
