# Operating Envelopes and Failure Mechanisms of Classical and Learned Path Planners

Code, trained models and evaluation logs for the paper *Operating Envelopes and Failure Mechanisms of Classical and Learned Path Planners*, submitted to *Robotics and Autonomous Systems*.

The paper measures the operating envelopes of six planners under one paired evaluation protocol:
- a goal-scaled **Artificial Potential Field (APF)** planner with a fixed rotational deflection, and **APF-D**, the same planner with heading momentum;
- the **Dynamic Window Approach**, plain (**DWA**) and with constant-velocity obstacle prediction (**DWA-CV**);
- three learned planners on one recurrent encoder: **ARC** (LSTM and DDPG), **TD3** and **SAC**, trained with and without moving obstacles, and TD3 with a pre-tanh penalty;
- an **arbiter** that runs SAC and hands control to APF-D (or DWA-CV) at thresholds derived from the measured envelopes.

## Key findings

- **Controlled conditions reorder the planners.** In distribution APF, DWA-CV and SAC succeed in 100, 94 and 83% of 50 paired dynamic environments. Restricted to the learned agent's sensing and actuation, APF retains 36 of its 50 dynamic successes, fewer than nine of 10 SAC policies.
- **APF stalls at stable equilibria of its own force field.** APF stalls on 32 of 100 unseen static layouts, 31 times at a stable equilibrium of the rotated, non-conservative field, around which the fixed-length step settles into a two-point oscillation. Heading momentum removes 20 of the 32 stalls, at a cost of up to eight of 50 dynamic successes at the highest speed ratio.
- **Deterministic actor-critic collapse is an optimiser trap.** The actor's tanh output saturates within the first 200 updates and the critic does not correct it. A pre-registered penalty on the pre-tanh output left none of 10 TD3 seeds collapsed, against five without it. The algorithm decides whether training collapses; the training distribution decides whether the policy survives moving obstacles.
- **Envelope-derived switching is a negative result.** The arbiter outperforms SAC alone (3,357 against 3,024 of 4,500 runs) but not its own fallback, APF-D on the same rays (3,990): envelopes measured per episode do not translate into per-state switching rules.

## The evaluation harness

`experiments/harness.py` controls the conditions that are usually left uncontrolled in planner comparisons:

| Factor | Control in the harness |
|---|---|
| Obstacle motion | `ObstacleTimeline(seed, layout)` integrates obstacle motion on a time grid shared by all planners; environment `s` is velocity seed `s`, so every planner faces the identical motion and results are paired by environment. |
| Speed ratio | `set_ground_speed(v)` sets the common agent speed; the obstacle tick is tied to the decision period, so outcomes depend on the speeds only through `r = 4.75 / v` (paper, Proposition 1; verified by `scripts/verify_section3.py scaleinv`). |
| Sensing and actuation | `apf_variants.run(..., sensing="rays", turn_cap=True, rate="arc")` runs APF under the learned agent's nine rays, turn-rate limit and decision rate. |
| Learned-planner variability | Every learned result is reported per training seed and per checkpoint (reward-selected, episode 300, best on the training layout), with each seed's collapse status. |

All planners return one result format (`outcome`, `steps`, `path_length`, `efficiency`, `min_clearance`, ...):

```python
import harness as H, apf_damped, dwa         # from experiments/
H.set_ground_speed(18.0)                                   # r = 0.26
tl = H.ObstacleTimeline(7)                                 # paired dynamic environment 7
H.run_apf(tl, planner_seed=1007)                           # APF
apf_damped.run(tl, 1007, momentum=0.85)                    # APF-D
dwa.run(tl, predict=True, weights={"alpha": 0.5, "beta": 0.2, "gamma": 0.2})   # DWA-CV
H.ARCPlanner("out/arc_models/sacdyn_seed0/models/arc_best.pth").run(tl)   # SAC (run from experiments/)
```

To add a planner, implement `run(timeline, ...)` returning the same dictionary and evaluate it on the frozen sets listed in `notes/seed_map.md`: motion seeds 0-49, the 100 random layouts from `random_layouts.make_layout(k)` with motion seed `10000 + k`, the second layout `H.LAYOUTS["urban"]`, and the speed and noise settings.

## Repository structure

```
├── experiments/        # planners, harness and the original evaluation scripts
│   ├── harness.py      #   shared protocol: timelines, ground speed, APF and learned-planner runners
│   ├── apf_variants.py #   APF under the learned agent's constraints
│   ├── apf_damped.py   #   APF-D (heading momentum) and the other damping candidates
│   ├── dwa.py          #   DWA and DWA-CV
│   ├── arbiter.py      #   the envelope-derived arbiter
│   ├── baselines.py    #   TD3 and SAC agents (optional pre-tanh penalty)
│   └── out/            #   original results (JSON) and trained models
├── scripts/            # revision experiments: one script per pre-registered step, job queue, statistics,
│                       #   make_numbers.py (every number in the paper) and make_figures.py
├── jobs/               # job files listing the runs of each experiment in order
├── results/v2/         # revision outputs, one folder per workstream and experiment, each with a manifest.json
├── prereg.md           # selection rules, recorded before the runs they govern
├── PROGRESS.md         # dated log of decisions and deviations
├── notes/              # seed map, scale-invariance audit, proofs
├── static_apf/, dynamic_apf/, apf_3d/   # APF subprojects (2D static, 2D dynamic, 3D)
├── arc_2d/, arc_3d/    # ARC subprojects (2D; 3D with TD3, HER, PER and curriculum)
└── docs/legacy/        # development documents, superseded by the paper
```

## Reproducing the paper

Original results: see [`experiments/README.md`](experiments/README.md). Revision results (damped APF, collapse mechanism, arbiter, statistics): each script in `scripts/` states its inputs and outputs in its docstring, and the job files in `jobs/` list the runs in order, for example

```bash
pip install numpy scipy torch shapely matplotlib tqdm tensorboard openpyxl psutil
python scripts/verify_section3.py reproduce          # stored results reproduce run for run
python scripts/jobqueue.py run --jobs jobs/c4.json   # APF-D candidates on the tuning sets (resumable)
python scripts/ws_c_c4_select.py                     # apply the pre-registered rule
python scripts/make_numbers.py && python scripts/make_figures.py
```

`scripts/jobqueue.py` runs one thread per job, skips finished jobs on restart and starts a job only while sufficient memory is free. Every revision output folder carries a `manifest.json` recording the git commit, configuration, seeds, start and end times, and whether training was running on the machine.

All results were produced on a single laptop (AMD Ryzen 7 5800HS, 16 GB RAM); all learned planners were trained on the CPU. The 3D ARC development in `arc_3d/` was run on Google Colab.

## Dependencies

Python 3.12, NumPy, SciPy, PyTorch, Shapely, Matplotlib, tqdm, TensorBoard (logging only), openpyxl and psutil. A CPU build of PyTorch is sufficient for every experiment in the paper.

## Citation

```bibtex
@unpublished{narayan2026envelopes,
  author = {Adi Narayan R S and Dev Sharma and Jenna Ann Sonali Sreshta Shetty and
            Anshul Sekuri and Akshat Richhariya and Muralikrishna S. N.},
  title  = {Operating Envelopes and Failure Mechanisms of Classical and Learned Path Planners},
  note   = {Submitted to Robotics and Autonomous Systems},
  year   = {2026}
}
```
