# Seed and layout map

Every random seed used by the experiments, where it is set, and what it controls. All
generators are `numpy.random.default_rng(seed)` unless stated. "Motion seed" = the seed of
`harness.ObstacleTimeline(seed, layout)`, which draws every obstacle's speed and direction.

## 1. Seeds in use (existing results; all frozen)

### Evaluation (frozen, never used for tuning)
| Set | Seeds | Set in | Notes |
|---|---|---|---|
| Custom environment, paired dynamic suite | motion 0-49 | `evaluate.py`, `test1_matched.py`, `dwa_eval.py` | 18 m/s |
| Speed sweep | motion 0-49 at every speed | `speed_sweep.py`, `dwa_eval.py` | 3, 4, 4.75, 6, 9, 12, 18, 27 m/s |
| Noise test | motion 0-49; noise generator 50000-50049 (`50_000 + s`) | `robustness_suite.py`, `dwa_eval.py` | levels L1-L3 |
| Second layout, paired dynamic suite | motion 0-49 on the second layout | `evaluate.py --layout urban`, `dwa_eval.py` | same seed values, different layout |
| 100 random layouts | generator 0-99 (`make_layout(k)`) | `random_layouts.py` | static and dynamic |
| 100 random layouts, dynamic | motion 10000-10099 (`10_000 + k`) | `robustness_suite.py`, `dwa_eval.py` | one motion seed per layout |
| 3D custom environment | 0-9 (perturbation and, for dynamic, 3D velocities from the same generator) | `run_apf3d.py` | |
| 3D ARC arena | perturbation 0-9; wind 10000-10009 (`10_000 + seed`) | `apf_in_arc3d.py` | separate world from the 2D layouts |

### Planner-internal generators (not motion)
| Use | Seeds | Set in |
|---|---|---|
| APF perturbation b, dynamic suite | 1000-1049 (`1000 + s`) | `evaluate.py`, `robustness_suite.py`, `speed_sweep.py`, `test1_matched.py` |
| APF perturbation b, random layouts dynamic | 20000-20099 (`20_000 + k`) | `robustness_suite.py` |
| APF perturbation b, static runs | 0 (static layouts), 0-9 (repeatability runs) | `robustness_suite.py`, `verify_apf_static.py`, `evaluate.py` |
| APF variant noise default (only when no noise seed is passed and noise > 0; never used in reported runs) | `planner_seed + 7_000_000` | `apf_variants.py` |
| Checkpoint evaluation: 10 random initial headings | 0 | `eval_checkpoints.py` |
| Gain sweep, theta sweep | 0 | `gain_sweep.py`, `verify_apf_static.py` |

### Training
| Use | Seeds | Set in |
|---|---|---|
| Learner seeds (Python `random`, `np.random.seed`, `torch.manual_seed`) | 0-4 (ARC, each layout), 0-9 (TD3 static, TD3 moving, SAC moving) | `train_arc_seed.py --seed N` |
| Exploring starts and exploration noise | drawn from the global generators seeded above | `arc_2d/code/train.py` |
| Moving-obstacle training: one fresh motion seed per rollout | `1_000_000 + randint(0, 1e9)`, i.e. [10^6, 10^6 + 10^9) | `dyn_train_env.py` |

### Tuning (existing)
| Use | Seeds | Set in |
|---|---|---|
| DWA / DWA-CV weight selection | motion 1000-1019 on the custom environment, plus the static custom environment | `dwa_tune.py` |

### Numerical overlaps between existing ranges (different purposes, recorded for completeness)
- Motion 1000-1019 (DWA tuning) and APF perturbation 1000-1049 share values but feed different generators (obstacle velocities vs. APF's perturbation b).
- Random-layout motion 10000-10099 and 3D wind 10000-10009 share values but belong to different worlds and generators.
- `planner_seed + 7_000_000` (APF variant noise default) falls inside the training motion range; it is never exercised in a reported run because every noisy run passes an explicit noise seed.

## 2. New held-out sets (for all tuning from now on)

Checked against every range above: no overlap in value with any evaluation, training or
existing tuning seed. Training motion seeds start at 10^6, so everything below is clear of them.

| Set | Seeds | Use |
|---|---|---|
| Held-out custom motion seeds | motion 500000-500049 on the custom environment | C4, D3 |
| APF perturbation on the held-out seeds | 501000-501049 (`501_000 + s`) | C4, D3 |
| Noise generator on the held-out seeds | 502000-502049 (`502_000 + s`) | D3 (2.5 m / 5 deg set) |
| Tuning layouts | generator 500000-500049 (`make_layout(500_000 + j)`), same generator and constraints as the evaluation layouts | C4, D3, static and dynamic |
| Tuning layouts, dynamic | motion 510000-510049 (`510_000 + j`) | C4, D3 |
| APF perturbation on tuning layouts | 0 (static, as for the evaluation layouts); 520000-520049 (`520_000 + j`, dynamic) | C4, D3 |

Any additional range must be added here before it is used.
